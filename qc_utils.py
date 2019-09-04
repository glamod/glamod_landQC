'''
qc_utils.py contains utility scripts to help with quality control tests
'''
import sys
import os
import pandas as pd
import numpy as np
import scipy.special

import configparser
import json
from scipy.optimize import least_squares


UNIT_DICT = {"temperature" : "degrees C", "dew_point_temperature" :  "degrees C", "wind_direction" :  "degrees", "wind_speed" : "meters per second", "sea_level_pressure" : "hPa hectopascals", "station_level_pressure" : "hPa hectopascals"}
MDI = -1.e30
MIN_NOBS = 100


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
# statistics
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

    qc_flags = np.core.defchararray.add(qc_flags, flags)

    return qc_flags # insert_flags


#************************************************************************
def populate_station(station, df, obs_var_list):
    """
    Convert Data Frame into internal station and obs_variable objects

    """
    
    for variable in obs_var_list:

        # make a variable
        this_var = Meteorological_Variable(variable, MDI, UNIT_DICT[variable], (float))
    
        # store the data
# TO DO
#        this_var.data = df[variable].to_numpy()
        indata = df[variable].fillna(MDI).to_numpy()
        this_var.data = np.ma.masked_where(indata == MDI, indata)
        this_var.data.fill_value = MDI

        # empty flag array
        this_var.flags = np.array(["" for i in range(len(this_var.data))])

        # and store
        setattr(station, variable, this_var)

    return # populate_station

#*********************************************
def IQR(data, percentile = 0.25):
    ''' Calculate the IQR of the data '''

    try:
        sorted_data = sorted(data.compressed())
    except AttributeError:
        # if not masked array
        sorted_data = sorted(data)
    
    n_data = len(sorted_data)

    quartile = int(round(percentile * n_data))
       
    return sorted_data[n_data - quartile] - sorted_data[quartile] # IQR
    
#*********************************************
def mean_absolute_deviation(data, median = False):    
    ''' Calculate the MAD of the data '''
    
    if median:
        mad = np.ma.mean(np.ma.abs(data - np.ma.median(data)))
        
    else:        
        mad = np.ma.mean(np.ma.abs(data - np.ma.mean(data)))

    return mad # mean_absolute_deviation

#*********************************************
def linear(X,p):
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
    err = ((Y-linear(X,p))**2.0)

    return err # residuals_linear

#*********************************************
def get_critical_values(indata, binmin = 0, binwidth = 1, plots = False, diagnostics = False, \
                        line_label = "", xlabel = "", title = "", old_threshold = 0):
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
        bins = np.arange(binmin, 2 * max(np.ceil(np.abs(indata))), binwidth)
        full_hist, full_edges = np.histogram(np.abs(indata), bins = bins)

        if len(full_hist) > 1:

            # use only the central section (as long as it's not just 2 bins)
            i = 0
            limit = 0
            while limit < 2:
                # count outwards until there is a zero-valued bin
                try:
                    limit = np.argwhere(full_hist == 0)[i][0]
                    i += 1
                except IndexError:
                    # no zero bins in this histogram
                    limit = len(full_hist)
                    break          

            # use this central section for fitting
            edges = full_edges[:limit]
            hist  = np.log10(full_hist[:limit])

            # remove inf's
            goods, = np.where(full_hist[:limit] != 0)
            edges = edges[goods]
            hist = hist[goods]

            # Working in log-yscale from hereon

            # a 10^-bx
            a = hist[np.argmax(hist)]
            b = 1

            p0 = np.array([a,b])
            result=least_squares(residuals_linear, p0, args=(hist, edges), max_nfev=10000, verbose=0, method="lm")

            fit = result.x

            fit_curve = linear(full_edges, fit)

            if fit[1] < 0:
                # negative slope as expected

                # where does *fit* fall below log10(0.1) = -1
                try:
                    fit_below_point1, = np.argwhere(fit_curve < -1)[0]

                    # find first empty bin after that
                    first_zero_bin, = np.argwhere(full_hist[fit_below_point1:] == 0)[0] + 1
                    threshold = binwidth * (binmin + fit_below_point1 + first_zero_bin)

                except IndexError:
                    # too shallow a decay - use default maximum
                    threshold = len(full_hist)

            else:
                # positive slope - likely malformed distribution.  Just retain all
                threshold = len(full_hist)

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
    plt.step(edges[1:], plot_hist, color='k', label=line_label, where="mid")
    plt.plot(edges, fit, 'b-', label="best fit")          
    
    plt.xlabel(xlabel)
    plt.ylabel("log10(Frequency)")
    
    # set y-lim to something sensible
    plt.ylim([-0.3, max(plot_hist)])
    plt.xlim([0, max(edges)])
    
    plt.axvline(threshold, c='r', label="threshold = {}".format(threshold))
    
    plt.legend(loc = "upper right")
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
        return np.subtract(*np.percentile(data.compressed(), [75, 25]))
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
def create_bins(data, width):

    bmin = np.floor(np.ma.min(data))
    bmax = np.ceil(np.ma.max(data))

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
    err = ((Y-skew_gaussian(X,p))**2.0)

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
    err = ((Y-gaussian(X,p))**2.0)

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
        mu=np.ma.mean(x)
    if sig == MDI:
        sig=np.ma.std(x)
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
