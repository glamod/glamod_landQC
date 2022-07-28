#!/usr/bin/env python
'''
qc_utils.py contains utility scripts to help with quality control tests
'''
import sys
import os
import configparser
import json
import pandas as pd
import numpy as np
import scipy.special

from scipy.optimize import least_squares

import setup


UNIT_DICT = {"temperature" : "degrees C", \
                 "dew_point_temperature" :  "degrees C", \
                 "wind_direction" :  "degrees", \
                 "wind_speed" : "meters per second", \
                 "sea_level_pressure" : "hPa hectopascals", \
                 "station_level_pressure" : "hPa hectopascals"}


QC_TESTS = {"o" : "Odd Cluster", "F" : "Frequent Value", "D" : "Distribution - Monthly", \
            "d" : "Distribution - all", "W" : "World Records", "K" : "Streaks", \
            "C" : "Climatological", "T" : "Timestamp", "S" : "Spike", "h" : "Humidity", \
            "V" : "Variance", "p" : "Pressure", "w" : "Winds", "L" : "Logic", "U" : "Diurnal", \
            "E" : "Clean Up", "N" : "Neighbour", "H" : "High Flag Rate"}


MDI = -1.e30


#*********************************************
# Process the Configuration File
#*********************************************

CONFIG_FILE = "./configuration.txt"

if not os.path.exists(os.path.join(os.path.dirname(__file__), CONFIG_FILE)):
    print("Configuration file missing - {}".format(os.path.join(os.path.dirname(__file__), CONFIG_FILE)))
    sys.exit()
else:
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), CONFIG_FILE)


config = configparser.ConfigParser()
config.read(CONFIG_FILE)

#*********************************************
# Statistics
MEAN = config.getboolean("STATISTICS", "mean")
MEDIAN = config.getboolean("STATISTICS", "median")
if MEAN == MEDIAN:
    print("Configuration file STATISTICS entry malformed. One of mean or median only")
    sys.exit


# MAD = 0.8 SD
# IQR = 1.3 SD
STDEV = config.getboolean("STATISTICS", "stdev")
IQR = config.getboolean("STATISTICS", "iqr")
MAD = config.getboolean("STATISTICS", "mad")
if sum([STDEV, MAD, IQR]) >= 2:
    print("Configuration file STATISTICS entry malformed. One of stdev, iqr, median only")
    sys.exit

#*********************************************
# Thresholds
DATA_COUNT_THRESHOLD = config.getint("THRESHOLDS", "min_data_count")
HIGH_FLAGGING = config.getfloat("THRESHOLDS", "high_flag_proportion")

# read in logic check list
LOGICFILE = os.path.join(os.path.dirname(__file__), config.get("FILES", "logic"))

#*********************************************
# Neighbour Checks
MAX_NEIGHBOUR_DISTANCE = config.getint("NEIGHBOURS", "max_distance")
MAX_NEIGHBOUR_VERTICAL_SEP = config.getint("NEIGHBOURS", "max_vertical_separation")
MAX_N_NEIGHBOURS = config.getint("NEIGHBOURS", "max_number")
NEIGHBOUR_FILE = config.get("NEIGHBOURS", "filename")
MIN_NEIGHBOURS = config.getint("NEIGHBOURS", "minimum_number")

#*********************************************
# Set up the Classes
#*********************************************
class Meteorological_Variable(object):
    '''
    Class for meteorological variable.  Initialised with metadata only
    '''
    
    def __init__(self, name, mdi, units, dtype):
        self.name = name
        self.mdi = mdi
        self.units = units
        self.dtype = dtype
        

    def __str__(self):     
        return "variable: {}".format(self.name)

    __repr__ = __str__
   
   
   
#*********************************************
class Station(object):
    '''
    Class for station
    '''
    
    def __init__(self, stn_id, lat, lon, elev):
        self.id = stn_id
        self.lat = lat
        self.lon = lon
        self.elev = elev
        
    def __str__(self):
        return "station {}, lat {}, lon {}, elevation {}".format(self.id, self.lat, self.lon, self.elev)
    
    __repr__ = __str__

    
#************************************************************************
# Subroutines
#************************************************************************
def get_station_list(restart_id="", end_id=""):
    """
    Read in station list file(s) and return dataframe

    :param str restart_id: which station to start on
    :param str end_id: which station to end on

    :returns: dataframe of station list
    """

    # process the station list
    station_list = pd.read_fwf(os.path.join(setup.SUBDAILY_MINGLE_DIR, setup.STATION_LIST), \
                                   widths=(11, 9, 10, 7, 35), header=None, names=("id", "latitude", "longitude", "elevation", "name"))

    # no longer necessary in November 2019 run, kept just in case
#    station_list2 = pd.read_fwf(os.path.join(setup.SUBDAILY_ROOT_DIR, "ghcnh-stations-2add.txt"), widths=(11, 9, 10, 7, 35), header=None)
#    station_list = station_list.append(station_list2, ignore_index=True)

    station_IDs = station_list.id

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = np.where(station_IDs == end_id)
        station_list = station_list.iloc[: endindex[0]+1]

    # and do the front
    if restart_id != "":
        startindex, = np.where(station_IDs == restart_id)
        station_list = station_list.iloc[startindex[0]:]

    return station_list.reset_index() # get_station_list

#************************************************************************
def read_qc_config(config_filename, section, field, islist=False):
    """
    Read the thresholds from relevant file
    
    :param str config_filename: filename of the config file
    :param str section: section heading (UPPERCASE)
    :param str field: field name
    """

    # Read config.ini file
    config_object = configparser.ConfigParser()
    config_object.read(config_filename)

    # Get the SECTION
    config_section = config_object[section]

    # Read the value
    if islist:
        value = json.loads(config_section[field])
    else:
        value = config_section[field]

    return value # read_qc_config

#************************************************************************
def write_qc_config(config_filename, section, field, value, diagnostics=False):
    """
    Write the thresholds into relevant file
    
    :param str config_filename: filename of the config file
    :param str section: section heading (UPPERCASE)
    :param str field: field name
    :param str value: value
    :param bool diagnostics: extra output
    """

    config_object = configparser.ConfigParser()

    if not os.path.isfile(config_filename):
        # make a new file
        cfgfile = open(config_filename, 'w')

        config_object.add_section(section)
        config_object.set(section, field, value)
        config_object.write(cfgfile)
        cfgfile.close()

    else:
        # file exists
        try:
            config_object.read(config_filename)
        except configparser.DuplicateSectionError:
            print("Malformed file {}".format(config_filename))
            sys.exit(1)
        

        try:
            config_section = config_object[section]
            config_section[field] = value
            # Write changes to file (replaces entire file)
            with open(config_filename, 'w') as conf:
                config_object.write(conf)
#            if diagnostics:
#                print("Updating file")

        except KeyError:
            # section doesn't exist
            config_object.add_section(section)
            config_object.set(section, field, value)
            # Write changes back to file
            with open(config_filename, 'w') as conf:
                config_object.write(conf)
#            if diagnostics:
#                print("Making section")

        except configparser.NoOptionError:
            # field doesn't exist
            config_object.set(section, field, value)
            # Write changes back to file
            with open(config_filename, 'w') as conf:
                config_object.write(conf)
#            if diagnostics:
#                print("Making field")

    return # write_qc_config

#************************************************************************
def insert_flags(qc_flags, flags):
    """
    Update QC flags with the new flags

    :param array qc_flags: string array of flags
    :param array flags: string array of flags
    """

    qc_flags = np.core.defchararray.add(qc_flags.astype(str), flags.astype(str))

    return qc_flags # insert_flags


#************************************************************************
def populate_station(station, df, obs_var_list, read_flags=False):
    """
    Convert Data Frame into internal station and obs_variable objects

    :param bool read_flags: read in already pre-existing flags
    """
    
    for variable in obs_var_list:

        # make a variable
        this_var = Meteorological_Variable(variable, MDI, UNIT_DICT[variable], (float))
    
        # store the data
#        this_var.data = df[variable].to_numpy()
        indata = df[variable].fillna(MDI).to_numpy()
        indata = indata.astype(float)
        this_var.data = np.ma.masked_where(indata == MDI, indata)
        if len(this_var.data.mask.shape) == 0:
            # single mask value, replace with arrage of True/False's
            if this_var.data.mask:
                # True
                this_var.data.mask = np.ones(this_var.data.shape)
            else:
                # False
                this_var.data.mask = np.zeros(this_var.data.shape)
            
        this_var.data.fill_value = MDI

        if read_flags:
            # change all empty values (else NaN) to blank
            this_var.flags = df["{}_QC_flag".format(variable)].fillna("").to_numpy()
        else:
            # empty flag array
            this_var.flags = np.array(["" for i in range(len(this_var.data))])

        # and store
        setattr(station, variable, this_var)

    return # populate_station

#*********************************************
def calculate_IQR(data, percentile=0.25):
    ''' Calculate the IQR of the data '''

    try:
        sorted_data = sorted(data.compressed())
    except AttributeError:
        # if not masked array
        sorted_data = sorted(data)
    
    n_data = len(sorted_data)

    quartile = int(round(percentile * n_data))
       
    return sorted_data[n_data - quartile] - sorted_data[quartile] # calculate_IQR
    
#*********************************************
def mean_absolute_deviation(data, median=False):    
    ''' Calculate the MAD of the data '''
    
    if median:
        mad = np.ma.mean(np.ma.abs(data - np.ma.median(data)))
        
    else:        
        mad = np.ma.mean(np.ma.abs(data - np.ma.mean(data)))

    return mad # mean_absolute_deviation

#*********************************************
def linear(X, p):
    '''
    decay function for line fitting
    p[0]=intercept
    p[1]=slope
    '''
    return p[1]*X + p[0] # linear

#*********************************************
def residuals_linear(p, Y, X):
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-linear(X, p))**2.0)

    return err # residuals_linear

#*********************************************
def get_critical_values(indata, binmin=0, binwidth=1, plots=False, diagnostics=False, \
                        line_label="", xlabel="", title="", old_threshold=0):
    """
    Plot histogram on log-y scale and fit 1/x decay curve to set threshold

    :param array indata: input data to bin up
    :param int binmin: minimum bin value
    :param int binwidth: bin width
    :param bool plots: do the plots
    :param bool diagnostics : do diagnostic outputs
    :param str line_label: label for plotted histogram
    :param str xlabel: label for x axis
    :param str title: plot title
 
    :returns:
       critical value

    """    
    if len(set(indata)) > 1:

        # set up the bins and make a histogram.  Use Absolute values
        max_bin = np.max([2 * max(np.ceil(np.abs(indata))), 10]) # so that have sufficient to fit to
        bins = np.arange(binmin, max_bin, binwidth)
        full_hist, full_edges = np.histogram(np.abs(indata), bins=bins)

        if len(full_hist) > 1:

            # use only the central section (as long as it's 5(10) or more bins)
            for limit_threshold in [5, 10]:
                i = 0
                n_zeros = 0
                limit = 0
                while limit < limit_threshold:
                    # count outwards until there is a zero-valued bin
                    try:
                        limit = np.argwhere(full_hist == 0)[i][0]
                        n_zeros += 1
                        i += 1
                    except IndexError:
                        # no zero bins in this histogram
                        limit = len(full_hist)
                        break

                if n_zeros >= 3 and limit == 5:
                    # check the next limit
                    pass
                else:
                    # got a decent enough central region
                    #   or extended limit isn't enough either, caught later
                    break

            if limit == 10 and n_zeros >= 7:
                # extended central bit is mainly zeros
                # can't continue
                threshold = max(indata) + binwidth

            else:

                # use this central section for fitting
                edges = full_edges[:limit]
                central_hist = full_hist[:limit]

                # remove inf's
                goods, = np.where(central_hist != 0)

                # if insufficient short streaks/small differences for centre of distribution
                if len(goods) < 2:
                    threshold = max(indata) + binwidth

                else:
                    hist = central_hist[goods]
                    edges = edges[goods]

                    # and take log10
                    hist = np.log10(hist)

                    # Working in log-yscale from hereon
                    # a 10^-bx
                    a = hist[np.argmax(hist)]
                    b = 1

                    p0 = np.array([a, b])
                    result = least_squares(residuals_linear, p0, args=(hist, edges), max_nfev=10000, verbose=0, method="lm")

                    fit = result.x

                    fit_curve = linear(full_edges, fit)

                    if fit[1] < 0:
                        # negative slope as expected

                        # where does *fit* fall below log10(0.1) = -1, then..
                        try:
                            fit_below_point1, = np.argwhere(fit_curve < -1)[0]

                            # find first empty bin after that
                            first_zero_bin, = np.argwhere(full_hist[fit_below_point1:] == 0)[0]
                            threshold = binwidth * (binmin + fit_below_point1 + first_zero_bin)

                        except IndexError:
                            # too shallow a decay - use default maximum.  Retains all data
                            threshold = len(full_hist)*binwidth

                    else:
                        # positive slope - likely malformed distribution.  Retains all data
                        threshold = len(full_hist)*binwidth

                    if plots:
                        plot_log_distribution(full_edges, full_hist, fit_curve, threshold, line_label, xlabel, title)

        else:
            threshold = max(indata) + binwidth

    elif len(set(indata)) == 1:
        threshold = max(indata) + binwidth

    else:
        # if no data, return 0+binwidth as the threshold to ensure a positive value
        threshold = np.copy(binwidth)
 
    return threshold # get_critical_values

#*********************************************
def plot_log_distribution(edges, hist, fit, threshold, line_label, xlabel, title):
    """
    Plot distribution on a log scale and show the fit

    """
    import matplotlib.pyplot as plt
    
    plt.clf()
    # stretch bars, so can run off below 0
    plot_hist = np.array([np.log10(x) if x != 0 else -1 for x in hist])
    plt.step(edges[1:], plot_hist, color='k', label=line_label, where="pre")
    plt.plot(edges, fit, 'b-', label="best fit")          
    
    plt.xlabel(xlabel)
    plt.ylabel("log10(Frequency)")
    
    # set y-lim to something sensible
    plt.ylim([-0.3, max(plot_hist)+0.5])
    plt.xlim([0, max(edges)])
    
    plt.axvline(threshold, c='r', label="threshold = {}".format(threshold))
    
    plt.legend(loc="upper right")
    plt.title(title)
       
    plt.show()

    return # plot_log_distribution


#*********************************************
def average(data):
    """
    Routine to wrap mean or median functions so can easily switch
    """

    if MEAN:
        return np.ma.mean(data)
    elif MEDIAN:
        return np.ma.median(data)

    # average

#*********************************************
def spread(data):
    """
    Routine to wrap st-dev, IQR or MAD functions so can easily switch
    """

    if STDEV:
        return np.ma.std(data)
    elif IQR:
        try:
            return np.subtract(*np.percentile(data.compressed(), [75, 25]))
        except AttributeError:
            return np.subtract(*np.percentile(data, [75, 25]))
            
    elif MAD:
        if MEDIAN:
            return np.ma.median(np.ma.abs(data - np.ma.median(data)))
        else:        
            return np.ma.mean(np.ma.abs(data - np.ma.mean(data)))

    # spread

#*********************************************
def winsorize(data, percent):
    """
    Replace data greater/less than upper/lower percentile with percentile value
    """
    
    for pct in [percent, 100-percent]:
        
        if pct < 50:
            percentile = np.percentile(data.compressed(), pct)
            locs = np.ma.where(data < percentile)
        else:
            percentile = np.percentile(data.compressed(), pct)
            locs = np.ma.where(data > percentile)
        
        data[locs] = percentile

    return data # winsorize

#************************************************************************
def create_bins(data, width, obs_var_name, anomalies=False):

    bmin = np.floor(np.ma.min(data))
    bmax = np.ceil(np.ma.max(data))

    try:
        bins = np.arange(bmin - (5*width), bmax + (5*width), width)
        return bins # create_bins
    except (MemoryError, ValueError):
        # wierd values (too small/negative or too high means lots of bins)
        # for INM00020460 Jan 2021
        import qc_tests.world_records as records

        # to ensure no overwriting of the obs variable name attribute
        if obs_var_name == "station_level_pressure":
            var_name = "sea_level_pressure"
            # hence use 500hPa as +/- search
        else:
            var_name = obs_var_name
            
        if var_name in ["station_level_pressure", "sea_level_pressure"]:
            pad = 500
        else:
            pad = 100
                        
        bmin = records.mins[var_name]["row"] - pad
        bmax = records.maxes[var_name]["row"] + pad

        if anomalies:
            # for INI0000VOMM June 2021
            # reset range to surround zero
            bmin = bmin - np.mean([bmin, bmax])
            bmax = bmax - np.mean([bmin, bmax])

        bins = np.arange(bmin - (5*width), bmax + (5*width), width)
        
        return bins # create_bins

#*********************************************
def gaussian(X, p):
    '''
    Gaussian function for line fitting
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    '''
    norm, mu, sig = p
    return (norm*(np.exp(-((X-mu)*(X-mu))/(2.0*sig*sig)))) # gaussian

#*********************************************
def skew_gaussian(X, p):
    '''
    Gaussian function for line fitting
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    p[3]=skew
    '''
    norm, mu, sig, skew = p
    return (norm*(np.exp(-((X-mu)*(X-mu))/(2.0*sig*sig)))) * \
        (1 + scipy.special.erf(skew*(X-mu)/(sig*np.sqrt(2)))) # skew_gaussian

#*********************************************
def residuals_skew_gaussian(p, Y, X):
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-skew_gaussian(X, p))**2.0)

    return err # residuals_skew_gaussian

#*********************************************
def invert_gaussian(Y, p):
    '''
    X value of Gaussian at given Y
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    '''
    norm, mu, sig = p
    return mu + (sig*np.sqrt(-2*np.log(Y/norm))) # invert_gaussian

#*********************************************
def residuals_gaussian(p, Y, X):
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-gaussian(X, p))**2.0)

    return err # residuals_gaussian

#*********************************************
def fit_gaussian(x, y, norm, mu=MDI, sig=MDI, skew=MDI):
    '''
    Fit a gaussian to the data provided
    Inputs:
      x - x-data
      y - y-data
      norm - norm
    Outputs:
      fit - array of [norm,mu,sigma,(skew)]
    '''
    if mu == MDI:
        mu = np.ma.mean(x)
    if sig == MDI:
        sig = np.ma.std(x)

    if sig == 0:
        # calculation of spread hasn't worked for some reason
        sig = 3.*np.unique(np.diff(x))[0]

    if skew == MDI:
        p0 = np.array([norm, mu, sig])
        result = least_squares(residuals_gaussian, p0, args=(y, x), max_nfev=10000, verbose=0, method="lm")
    else:
        p0 = np.array([norm, mu, sig, skew])
        result = least_squares(residuals_skew_gaussian, p0, args=(y, x), max_nfev=10000, verbose=0, method="lm")
    return result.x # fit_gaussian

#************************************************************************
def find_gap(hist, bins, threshold, gap_size, upwards=True):
    '''
    Walk the bins of the distribution to find a gap and return where it starts
   
    :param array hist: histogram values
    :param array bins: bin values
    :param flt threshold: limiting value
    :param int gap_size: gap size to record
    :param bool upwards: for positive part of x-axis
    :returns:
        flt: gap_start
    '''

    # start in the centre
    start = np.argmax(hist)
       
    n = 0
    gap_length = 0
    gap_start = 0
    while True:
        # if bin is zero - could be a gap
        if hist[start + n] == 0:
            gap_length += 1
            if gap_start == 0:
                # plus 1 to get upper bin boundary
                if (upwards and bins[start + n + 1] >= threshold):
                    gap_start = bins[start + n + 1]
                elif (not upwards and bins[start + n] <= threshold):
                    gap_start = bins[start + n]
                
        # bin has value
        else:
            # gap too short
            if gap_length < gap_size:
                gap_length = 0
                gap_start = 0
                
            # found a gap
            elif gap_length >= gap_size and gap_start != 0:
                break
        
        # increment counters
        if upwards:
            n += 1
        else:
            n -= 1

        # escape if gone off the end of the distribution
        if (start + n == len(hist) - 1) or (start + n == 0):
            gap_start = 0
            break

    return gap_start # find_gap

#*********************************************
def reporting_accuracy(indata, winddir=False, plots=False):
    '''
    Uses histogram of remainders to look for special values

    :param array indata: masked array
    :param bool winddir: true if processing wind directions
    :param bool plots: make plots (winddir only)

    :returns: resolution - reporting accuracy (resolution) of data
    '''
    
    
    good_values = indata.compressed()

    resolution = -1
    if winddir:
        # 360/36/16/8/ compass points ==> 1/10/22.5/45/90 deg resolution
        if len(good_values) > 0:

            hist, binEdges = np.histogram(good_values, bins=np.arange(0, 362, 1))

            # normalise
            hist = hist / float(sum(hist))

            #
            if sum(hist[np.arange(90, 360+90, 90)]) >= 0.6:
                resolution = 90
            elif sum(hist[np.arange(45, 360+45, 45)]) >= 0.6:
                resolution = 45
            elif sum(hist[np.round(0.1 + np.arange(22.5, 360+22.5, 22.5)).astype("int")]) >= 0.6:
                # added 0.1 because of floating point errors!
                resolution = 22
            elif sum(hist[np.arange(10, 360+10, 10)]) >= 0.6:
                resolution = 10
            else:
                resolution = 1

            print("Wind dir resolution = {} degrees".format(resolution))
            if plots:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.hist(good_values, bins=np.arange(0, 362, 1))
                plt.show()
        
    else:
        if len(good_values) > 0:

            remainders = np.abs(good_values) - np.floor(np.abs(good_values))

            hist, binEdges = np.histogram(remainders, bins=np.arange(-0.05, 1.05, 0.1))

            # normalise
            hist = hist / float(sum(hist))

            if hist[0] >= 0.3:
                if hist[5] >= 0.15:
                    resolution = 0.5
                else:
                    resolution = 1.0
            else:
                resolution = 0.1

    return resolution # reporting_accuracy

#*********************************************
def reporting_frequency(intimes, inobs):
    '''
    Uses histogram of remainders to look for special values

    Works on hourly and above or minute data separately

    :param array intimes: array of panda datetimes
    :param array inobs: masked array
    :returns: frequency - reporting frequency of data (minutes)
    '''
        
    masked_times = np.ma.masked_array(intimes, mask=inobs.mask)

    frequency = -1
    if len(masked_times) > 0:

        difference_series = np.ma.diff(masked_times)/np.timedelta64(1, "m")

        if np.unique(difference_series)[0] >= 60:
            # then most likely hourly or beyond
            
            difference_series = difference_series/60.

            hist, binEdges = np.histogram(difference_series, bins=np.arange(1, 25, 1), density=True)
            # 1,2,3,6 hours
            if hist[0] >= 0.5:
                frequency = 60
            elif hist[1] >= 0.5:
                frequency = 120
            elif hist[2] >= 0.5:
                frequency = 180
            elif hist[3] >= 0.5:
                frequency = 240
            elif hist[5] >= 0.5:
                frequency = 360
            else:
                frequency = 1440

        else:
            # have to think about minutes
            hist, binEdges = np.histogram(difference_series, bins=np.arange(1, 60, 1), density=True)
            # 1,5,10 minutes
            if hist[0] >= 0.5:
                frequency = 1
            elif hist[4] >= 0.5:
                frequency = 5
            elif hist[9] >= 0.5:
                frequency = 10
            else:
                frequency = 60
       
    return frequency # reporting_frequency

#*********************************************
#DEPRECATED - now in a test
def high_flagging(station):
    """
    Check flags for each observational variable, and return True if any 
    has too large a proportion flagged
    
    :param Station station: station object
    
    :returns: bool
    """
    bad = False

    for ov in setup.obs_var_list:

        obs_var = getattr(station, ov)

        obs_locs, = np.where(obs_var.data.mask == False)

        if obs_locs.shape[0] > 10 * DATA_COUNT_THRESHOLD:
            # require sufficient observations to make a flagged fraction useful.

            flags = obs_var.flags

            flagged, = np.where(flags[obs_locs] != "")

            if flagged.shape[0] / obs_locs.shape[0] > HIGH_FLAGGING:
                bad = True
                print("{} flagging rate of {:5.1f}%".format(obs_var.name, \
                                                                100*(flagged.shape[0] / obs_locs.shape[0])))
                break

    return bad # high_flagging


#************************************************************************
def find_country_code(lat, lon):
    """
    Use reverse Geocoder to find closest city to each station, and hence
    find the country code.

    :param float lat: latitude
    :param float lon: longitude

    :returns: [str] country_code
    """
    import reverse_geocoder as rg
    results = rg.search((lat, lon))
    country = results[0]['cc']

    return country # find_country_code

#************************************************************************
def find_continent(country_code):
    """
    Use ISO country list to find continent from country_code.

    :param str country_code: ISO standard country code

    :returns: [str] continent
    """
 
    # prepare look up
    with open('iso_country_codes.json', 'r') as infile:
        iso_codes = json.load(infile)

    concord = {}
    for entry in iso_codes:
        concord[entry["Code"]] = entry["continent"]

    return concord[country_code]
