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
import pathlib
import itertools
import logging

from scipy.optimize import least_squares

import setup


UNIT_DICT = {"temperature" : "degrees C", \
             "dew_point_temperature" :  "degrees C", \
             "wind_direction" :  "degrees", \
             "wind_speed" : "meters per second", \
             "sea_level_pressure" : "hPa hectopascals", \
             "station_level_pressure" : "hPa hectopascals"}

# Letters for flags which should exclude data
# Numbers for information flags:
#   the data are valid, but not necessarily adhering to conventions
QC_TESTS = {"C" : "Climatological",
            "D" : "Distribution - Monthly",
            "E" : "Clean Up",
            "F" : "Frequent Value",
            "H" : "High Flag Rate",
            "K" : "Repeating Streaks",
            "L" : "Logic",
            "N" : "Neighbour",
            "S" : "Spike",
            "T" : "Timestamp",
            "U" : "Diurnal",
            "V" : "Variance",
            "W" : "World Records",
            "d" : "Distribution - all",
            "h" : "Humidity",
            "n" : "Precision",
            "o" : "Odd Cluster",
            "p" : "Pressure",
            "w" : "Winds",
            "x" : "Excess streak proportion",
            "y" : "Repeated Day streaks",
            "1" : "Wind logical - calm, masked direction",
            "2" : "Timestamp - identical observation values",
            }


MDI = -1.e30
FIRST_YEAR = 1700
# Can have these values for station elevations which are allowed, but indicate missing
#   information.  Blank entries also allowed, but others which do not make sense
#   are caught by logic check.  Need to escape these for e.g. pressure checks
ALLOWED_MISSING_ELEVATIONS = ["-999", "9999"]

# These data are retained and processed by the QC tests.  All others are not.
WIND_MEASUREMENT_CODES = ["", "N-Normal", "C-Calm", "V-Variable", "9-Missing"]


#*********************************************
# Process the Configuration File
#*********************************************

CONFIG_FILE = "./configuration.txt"

if not os.path.exists(os.path.join(os.path.dirname(__file__), CONFIG_FILE)):
    print(f"Configuration file missing - {os.path.join(os.path.dirname(__file__), CONFIG_FILE)}")
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
        self.data = None


    def __str__(self):
        return f"variable: {self.name}"

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
        return f"station {self.id}, lat {self.lat}, lon {self.lon}, elevation {self.elev}"

    __repr__ = __str__


#************************************************************************
# Subroutines
#************************************************************************
def get_station_list(restart_id: str = "", end_id: str = "") -> pd.DataFrame:
    """
    Read in station list file(s) and return dataframe

    :param str restart_id: which station to start on
    :param str end_id: which station to end on

    :returns: dataframe of station list
    """
    # Test if station list fixed-width format or comma separated
    #  [Initially supplied nonFWF format for Release 8 processing]
    fwf = True
    with open(setup.STATION_LIST, "r") as infile:
        lines = infile.readlines()
        for row in lines:
            # The non FWF version had double quotes present around the station names
            #    but was space separated, so can use that to determine if it's FWF or not
            if row.find('"') != -1:
                fwf = False

    # process the station list
    if fwf:
        # Fixed width format
        # If station-ID has format "ID-START-END" then width is 29
        station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5),
                                header=None, names=("id", "latitude", "longitude", "elevation", "state",
                                                    "name", "wmo"))
    else:
        # Comma separated
        station_list = pd.read_csv(setup.STATION_LIST, delim_whitespace=True,
                                header=None, names=("id", "latitude", "longitude", "elevation", "name"))
        # add extra columns (despite being empty) so these are available to later stages
        for newcol in ["state", "wmo"]:
            station_list[newcol] = ["" for i in range(len(station_list))]

    # fill empty entries (default NaN) with blank strings
    station_list = station_list.fillna("")

    station_IDs = station_list.id

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = np.where(station_IDs == end_id)
        station_list = station_list.iloc[: endindex[0]+1]

    # and do the front
    if restart_id != "":
        startindex, = np.where(station_IDs == restart_id)
        station_list = station_list.iloc[startindex[0]:]

    return station_list.reset_index(drop=True) # get_station_list


#************************************************************************
def insert_flags(qc_flags: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """
    Update QC flags with the new flags

    :param array qc_flags: string array of flags
    :param array flags: string array of flags
    """

    qc_flags = np.core.defchararray.add(qc_flags.astype(str), flags.astype(str))

    return qc_flags # insert_flags


#************************************************************************
def populate_station(station: Station, df: pd.DataFrame, obs_var_list: list, read_flags: bool = False) -> None:
    """
    Convert Data Frame into internal station and obs_variable objects

    :param Station station: station object to hold information
    :param DataFrame df: dataframe of input data
    :param list obs_var_list: list of observed variables
    :param bool read_flags: read in already pre-existing flags
    """

    for variable in obs_var_list:

        # make a variable
        this_var = Meteorological_Variable(variable, MDI, UNIT_DICT[variable], (float))

        # store the data
        indata = df[variable].fillna(MDI).to_numpy()
        indata = indata.astype(float)

        # For wind direction and speed only, account for some measurement flags
        #  Mask data in the Met_Var object used for the tests, but leave dataframe
        #  unaffected.
        if variable in ["wind_direction", "wind_speed"]:
            m_code = df[f"{variable}_Measurement_Code"]

            # Build up the mask
            for c, code in enumerate(WIND_MEASUREMENT_CODES):
                if code == "":
                    # Empty flags converted to NaNs on reading
                    code = float("NaN")
                    if c == 0:
                        mask = (m_code == code)
                    else:
                        mask = (m_code == code) | mask
                else:
                    # Doing string comparison
                    if c == 0:
                        # Initialise
                        mask = (m_code.str.startswith(code))
                    else:
                        # Combine using or
                        #   e.g. if code = "N-Normal" or "C-Calm" or "" set True
                        mask = (m_code.str.startswith(code)) | mask

            # invert mask and set to missing
            indata[~mask] = MDI

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
            this_var.flags = df[f"{variable}_QC_flag"].fillna("").to_numpy()
        else:
            # empty flag array
            this_var.flags = np.array(["" for i in range(len(this_var.data))])

        # and store
        setattr(station, variable, this_var)

    return # populate_station

#*********************************************
def calculate_IQR(data: np.ndarray, percentile: float = 0.25) -> float:
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
def mean_absolute_deviation(data: np.ndarray, median: bool = False) -> float:
    ''' Calculate the MAD of the data '''

    if median:
        mad = np.ma.mean(np.ma.abs(data - np.ma.median(data)))

    else:
        mad = np.ma.mean(np.ma.abs(data - np.ma.mean(data)))

    return mad # mean_absolute_deviation

#*********************************************
def linear(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    decay function for line fitting
    p[0]=intercept
    p[1]=slope
    '''
    return p[1]*X + p[0] # linear

#*********************************************
def residuals_linear(p: np.ndarray, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-linear(X, p))**2.0)

    return err # residuals_linear


#*********************************************
def gcv_calculate_binmax(indata: np.ndarray, binmin: float, binwidth: float) -> float:
    """
    Determine the appropriate largest bin to use.

    :param array indata: input data to bin up
    :param float binmin: minimum bin value
    :param float binwidth: bin width

    :returns: binmax (float)
    """
    logger = logging.getLogger(__name__)

    MAX_N_BINS = 20000
    # so that have sufficient x-bins to fit to
    if binwidth < 0.1:
        binmax = np.max([2 * max(np.ceil(np.abs(indata))), 1])
    else:
        binmax = np.max([2 * max(np.ceil(np.abs(indata))), 10])

    # if too big, then adjust
    if (binmax - binmin)/binwidth > MAX_N_BINS:
        # too many bins, will run out of memory
        logger.warning(f" Too many bins requested: {binmin} to {binmax} in steps of {binwidth}")
        binmax = binmin + (MAX_N_BINS * binwidth)
        logger.warning(f" Setting binmax to {binmax}")

    return binmax  #gcv_calculate_binmax


#*********************************************
def gcv_zeros_in_central_section(histogram: np.ndarray, inner_n: int) -> int:
    """
    Helper routine for get_critical_values() ["gcv"] to determine if, for a distribution
    with multiple peaks, the central section is sufficiently big.  When fitting a -x line
    in get_critical_values(), this may not work as intended if many of the bins close to x=0 have
    y=0.  This routine counts the number of zero-valued bins within the inner_n bins.

    :param array histogram: histogram of values to assess
    :param int inner_n: how many of the inner bins to assess

    :returns: n_zeros   how many n_zeros within limit bins of the centre
    """

    if len(np.nonzero(histogram == 0)[0]) == 0:
        # No zero bins, so central section is the whole histogram
        return 0

    # Use only the central section (as long as it's 5(10) or more bins)
    n_zeros = 0
    index = 0
    while index < inner_n:
        # if not gone beyond the end of the histogram
        if index >= len(histogram):
            break
        # Count outwards until there is a zero-valued bin
        if histogram[index] == 0:
            n_zeros += 1
        index += 1

    return n_zeros  # gcv_zeros_in_central_section


#*********************************************
def gcv_linear_fit_to_log_histogram(histogram: np.array, bins: np.array) -> np.array:
    """
    Take the log10 of the histogram values, and fit a linear -x line (10^-x in reality)

    :param array histogram: the histogram values to fit
    :param array bins: the histogram bins

    :returns: array of fit parameters of (norm, slope)
    """

    # and take log10
    histogram = np.log10(histogram)

    # Working in log-yscale
    # a 10^bx, expecting b to be negative
    a = histogram[np.argmax(histogram)]
    b = 1

    p0 = np.array([a, b])
    result = least_squares(residuals_linear, p0, args=(histogram, bins), max_nfev=10000, verbose=0, method="lm")

    return result.x


#*********************************************
def get_critical_values(indata: np.ndarray, binmin: float = 0, binwidth: float = 1,
                        plots: bool = False, diagnostics: bool = False,
                        line_label: str = "", xlabel: str = "", title: str = "") -> float:

    """
    Plot histogram on log-y scale and fit -x line (equivalent to
    exp(-x) decay curve) to set threshold

    :param array indata: input data to bin up
    :param float binmin: minimum bin value
    :param float binwidth: bin width
    :param bool plots: do the plots
    :param bool diagnostics : do diagnostic outputs
    :param str line_label: label for plotted histogram
    :param str xlabel: label for x axis
    :param str title: plot title

    :returns:
       float critical value
    """
    if len(indata) == 0:
        # If no data, return 0+binwidth as the threshold to ensure a positive value
        threshold = 0+binwidth
        return threshold

    # for spike, need absolute values
    default_threshold = np.max(np.abs(indata)) + binwidth

    if len(set(indata)) == 1:
        # Single datapoint, so set threshold above this
        return default_threshold

    # Or there is data to process, let's go
    # set up the bins and make a histogram.  Use Absolute values
    binmax = gcv_calculate_binmax(indata, binmin, binwidth)
    bins = np.arange(binmin, binmax, binwidth)
    full_hist, full_edges = np.histogram(np.abs(indata), bins=bins)

    if len(full_hist) <= 1:
        # single bin, so just take the max of the data
        return default_threshold

    # Check if the first 5(10) bins have sufficient data
    n_zeros = gcv_zeros_in_central_section(full_hist, 5)
    if n_zeros >= 3:
        # Note: although cannot have streaks length < 2, this is handled
        #       via the binmin argument (set to 2 in humidity DPD and streaks)
        if len(full_hist) > 5:
            n_zeros = gcv_zeros_in_central_section(full_hist, 10)
            if n_zeros >= 6:
                # Extended central bit is mainly zeros
                # can't continue, set threshold to exceed data
                return default_threshold
        else:
            # Extended central bit is mainly zeros
            # can't continue, set threshold to exceed data
            return default_threshold

    # Use this central section for fitting
    #  Avoids risk of secondary populations in the distribution affecting the fit
    #  Allow for well behaved (i.e. without zero value bins) distributions
    #    and take distance to the first zero bin, or 10, whichever larger
    first_zero_bin, = np.argwhere(full_hist[0:] == 0)[0]
    central = np.max([10, first_zero_bin])
    edges = full_edges[:central]
    central_hist = full_hist[:central]

    # Remove zeros (turn into infs in log-space)
    goods, = np.nonzero(central_hist != 0)
    hist = central_hist[goods]
    edges = edges[goods]

    # Get the curve, and the best fit points
    fit = gcv_linear_fit_to_log_histogram(hist, edges)
    fit_curve = linear(full_edges, fit)


    if fit[1] < 0:
        # negative slope as expected

        # where does *fit* fall below log10(0.1) = -1, then..
        try:
            fit_below_point1, = np.argwhere(fit_curve < -1)[0]
            # find first empty bin after that
            # TODO: think about using 2 empty bins?
            first_zero_bin, = np.argwhere(full_hist[fit_below_point1:] == 0)[0]
            threshold = binwidth * (binmin + fit_below_point1 + first_zero_bin)
            if isinstance(threshold, np.integer):
                # JSON encoder can't cope with np.int64 objects
                threshold = int(threshold)

        except IndexError:
            # Too shallow a decay - use default maximum.  Retains all data
            #   If there were a value much higher, then because a negative
            #   slope the above snippet should run, rather than this one.
            threshold = default_threshold

    else:
        # Positive slope - likely malformed distribution.  Retains all data
        #    The test won't work well given the fit, so just take the data max.
        threshold = default_threshold

    if plots:
        plot_log_distribution(full_edges, full_hist, fit_curve, threshold, line_label, xlabel, title)

    return threshold # get_critical_values


#*********************************************
def plot_log_distribution(edges: np.ndarray, hist: np.ndarray, fit: np.ndarray, threshold: float, line_label: str, xlabel: str, title: str) -> None:
    """
    Plot distribution on a log scale and show the fit

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _, ax = plt.subplots()

    # stretch bars, so can run off below 0
#    plot_hist = np.array([np.log10(x) if x != 0 else -1 for x in hist])
#    plt.step(edges[1:], plot_hist, color='k', label=line_label, where="pre")

    # set values == 0 to be 0.01, so can plot on a log plot
    plot_hist = np.array([x if x != 0 else 0.01 for x in hist])
    plt.step(edges[:-1], plot_hist, color='k', label=line_label, where="mid")

    # convert the fit in log space to actuals
    fit = [10**i for i in fit]
    plt.plot(edges, fit, 'b-', label="best fit")

    plt.xlabel(xlabel)
    plt.ylabel("Frequency (logscale))")

    # set y-lim to something sensible in actual space
    plt.ylim([-1.3, max(plot_hist)+0.5])
    plt.ylim([0.01, max(plot_hist)*3])
    plt.xlim([0, max(edges)])

    plt.axvline(threshold, c='r', label=f"threshold = {threshold:.2f}")

    plt.legend(loc="upper right")
    plt.title(title)
    plt.yscale("log")

    # sort axes formats
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    if max(edges) > 2:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    else:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    plt.show()

    return # plot_log_distribution


#*********************************************
def average(data: np.ndarray) -> float:
    """
    Routine to wrap mean or median functions so can easily switch
    """

    if MEAN:
        return np.ma.mean(data)
    elif MEDIAN:
        return np.ma.median(data)

    # average

#*********************************************
def spread(data: np.ndarray) -> float:
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
def winsorize(data: np.ndarray, percent: float) -> np.ndarray:
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
def create_bins(data: np.ndarray, width: float, obs_var_name: str, anomalies: bool = False):

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
def gaussian(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    '''
    Gaussian function for line fitting
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    '''
    norm, mu, sig = p
    return (norm*(np.exp(-((X-mu)*(X-mu))/(2.0*sig*sig)))) # gaussian

#*********************************************
def skew_gaussian(X: np.ndarray, p: np.ndarray) -> np.ndarray:
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
def residuals_skew_gaussian(p: np.ndarray, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-skew_gaussian(X, p))**2.0)

    return err # residuals_skew_gaussian

#*********************************************
def invert_gaussian(Y: float, p: float) -> float:
    '''
    X value of Gaussian at given Y
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    '''
    norm, mu, sig = p
    return mu + (sig*np.sqrt(-2*np.log(Y/norm))) # invert_gaussian

#*********************************************
def residuals_gaussian(p: np.ndarray, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
    Least squared residuals from linear trend
    '''
    err = ((Y-gaussian(X, p))**2.0)

    return err # residuals_gaussian

#*********************************************
def fit_gaussian(x: np.ndarray, y: np.ndarray,
                 norm: float, mu: float = MDI,
                 sig: float = MDI, skew: float = MDI) -> np.ndarray:
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

    if np.isnan(skew):
        # calculation of skew hasn't worked for some reason (e.g. no variation in values, all==0)
        skew = 0

    # call the appropriate fitting function and routine
    if skew == MDI:
        p0 = np.array([norm, mu, sig])
        result = least_squares(residuals_gaussian, p0, args=(y, x), max_nfev=10000, verbose=0, method="trf", jac="3-point")
    else:
        p0 = np.array([norm, mu, sig, skew])
        result = least_squares(residuals_skew_gaussian, p0, args=(y, x), max_nfev=10000, verbose=0, method="trf", jac="3-point")
    return result.x # fit_gaussian

#************************************************************************
def find_gap(hist: np.ndarray, bins: np.ndarray, threshold: float, gap_size: int, upwards: bool = True) -> float:
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
def reporting_accuracy(indata: np.ndarray, winddir: bool = False, plots: bool = False) -> float:
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

            print(f"Wind dir resolution = {resolution} degrees")
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

            if plots:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.hist(remainders, bins=np.arange(-0.05, 1.05, 0.1), density=True)
                plt.show()

    return resolution # reporting_accuracy

#*********************************************
def reporting_frequency(intimes: np.ndarray, inobs: np.ndarray) -> float:
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
def high_flagging(station: Station) -> bool:
    """
    Check flags for each observational variable, and return True if any
    has too large a proportion flagged

    :param Station station: station object

    :returns: bool
    """
    bad = False

    for ov in setup.obs_var_list:

        obs_var = getattr(station, ov)

        obs_locs, = np.nonzero(obs_var.data.mask == False)

        if obs_locs.shape[0] > 10 * DATA_COUNT_THRESHOLD:
            # require sufficient observations to make a flagged fraction useful.

            flags = obs_var.flags

            flagged, = np.nonzero(flags[obs_locs] != "")

            if flagged.shape[0] / obs_locs.shape[0] > HIGH_FLAGGING:
                bad = True
                print(f"{obs_var.name} flagging rate of {100*(flagged.shape[0] / obs_locs.shape[0]):5.1f}%")
                break

    return bad # high_flagging


#************************************************************************
def find_country_code(lat: float, lon: float) -> str:
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
def find_continent(country_code: str) -> str:
    """
    Use ISO country list to find continent from country_code.

    :param str country_code: ISO standard country code

    :returns: [str] continent
    """

    # as maybe run from another directory, get the right path
    cwd = pathlib.Path(__file__).parent.absolute()
    # prepare look up
    with open(f'{cwd}/iso_country_codes.json', 'r') as infile:
        iso_codes = json.load(infile)

    concord = {}
    for entry in iso_codes:
        concord[entry["Code"]] = entry["continent"]

    return concord[country_code]

#************************************************************************
def prepare_data_repeating_streak(data: np.ndarray, diff:int = 0,
                                  plots:bool = False, diagnostics:bool = False) -> tuple[np.array,
                                                                                         np.array,
                                                                                         np.array]:
    """
    Prepare the data for repeating streaks

    :param np.array data: data to assess
    :param int diff: difference to look for (0 in streaks of data - i.e. same values
                                             1 in streaks of indices - i.e. adjacent locations)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: tuple(array, array, array)

    array of the streak lengths in temporal order
    array of the grouped differences (difference, count)
    array of the streak locations [in grouped differences, so need double expansion]
    """

    # want locations where first differences are zero
    #   does not change if time or instrumental resolution changes.
    #   if there is zero difference between one obs or index and the next, that's the main thing
    value_diffs = np.ma.diff(data)

    # group the differences
    #     array of (value_diff, count) pairs
    #     Inspired by https://stackoverflow.com/a/58222158
    grouped_diffs = np.array([[g[0], len(list(g[1]))] for g in itertools.groupby(value_diffs)])

    # all streak lengths
    streaks, = np.nonzero(grouped_diffs[:, 0] == diff)
    repeated_streak_lengths = grouped_diffs[streaks, 1] + 1

    return repeated_streak_lengths, grouped_diffs, streaks # prepare_data_repeating_streak


#************************************************************************
def custom_logger(logfile: str):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Remove any current handlers; the handlers persist within the same
    # Python session, so ensure the root logger is 'clean' every time
    # this function is called. Note that using logger.removeHandler()
    # doesn't work reliably
    logger.handlers = []

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create file handler to capture all output
    fh = logging.FileHandler(logfile, "w")
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    logconsole_format = logging.Formatter('%(levelname)-8s %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(logconsole_format)

    logfile_format = logging.Formatter('%(asctime)s %(module)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(logfile_format)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
