#!/usr/bin/env python
'''
qc_utils.py contains utility scripts to help with QC checks
'''
import numpy as np
import pandas as pd
import scipy.special
import itertools
import logging

from scipy.optimize import least_squares

import qc_tests.world_records as records
import utils

MAX_N_BINS = 20000

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

    if np.count_nonzero(histogram == 0) == 0:
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
def plot_log_distribution(edges: np.ndarray, hist: np.ndarray,
                          fit: np.ndarray, threshold: float,
                          line_label: str, xlabel: str,
                          title: str) -> None:  # pragma: no cover
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

    # plot_log_distribution


#*********************************************
def average(data: np.ndarray) -> float:
    """
    Routine to wrap mean or median functions so can easily switch
    """

    if utils.MEAN:
        return np.ma.mean(data)
    elif utils.MEDIAN:
        return np.ma.median(data)

    # average

#*********************************************
def spread(data: np.ndarray) -> float:
    """
    Routine to wrap st-dev, IQR or MAD functions so can easily switch
    """

    if utils.STDEV:
        return np.ma.std(data)
    elif utils.IQR:
        try:
            return np.subtract(*np.percentile(data.compressed(), [75, 25]))
        except AttributeError:
            return np.subtract(*np.percentile(data, [75, 25]))

    elif utils.MAD:
        if utils.MEDIAN:
            return np.ma.median(np.ma.abs(data - np.ma.median(data)))
        else:
            return np.ma.mean(np.ma.abs(data - np.ma.mean(data)))

    # spread

#*********************************************
def winsorize(data: np.ma.MaskedArray,
              percent: float) -> np.ma.MaskedArray:
    """Replace data greater/less than upper/lower percentile with percentile value

    Parameters
    ----------
    data : np.ma.MaskedArray
        input data
    percent : float
        percentile at which to cut

    Returns
    -------
    np.ma.MaskedARray
        updated data
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
def create_bins(data: np.ndarray, width: float,
                obs_var_name: str, anomalies: bool = False) -> np.ndarray:
    """Create a number of bins from the data given the width

    Parameters
    ----------
    data : np.ndarray
        Data from which to determine reasonable bins
    width : float
        Width of bins
    obs_var_name : str
        Name of variable [to be able to check pressure types]
    anomalies : bool, optional
        Select if values are anomalies, by default False

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """

    logger = logging.getLogger(__name__)
    # get the lowest and highest possible values (int) from data
    bmin = np.floor(np.ma.min(data))
    bmax = np.ceil(np.ma.max(data))

    # default behaviour, use the width and pad a bit to give some clear space
    bins = np.arange(bmin - (5*width), bmax + (5*width), width)

    if len(bins) > MAX_N_BINS:
        logger.warning(f" Too many bins requested: {bmin} to {bmax} in steps of {width}")
        # Too many to be reasonably handled.
        # Likely because of erroneous values causing too high max/low min
        # to ensure no overwriting of the obs variable name attribute

        var_name = f"{obs_var_name}"
        if obs_var_name == "station_level_pressure":
            var_name = "sea_level_pressure"

        pad = 100
        if "pressure" in obs_var_name:
            pad = 500  # hpa

        # Using Rest Of World (ROW) to get total range
        bmin = np.floor(records.mins[var_name]["row"]) - pad
        bmax = np.ceil(records.maxes[var_name]["row"]) + pad

        if anomalies:
            # for INI0000VOMM June 2021
            # reset range to surround zero
            bmin = bmin - np.mean([bmin, bmax])
            bmax = bmax - np.mean([bmin, bmax])

        logger.warning(f" Setting binmax range to {bmin} to {bmax}")
        bins = np.arange(bmin - (5*width), bmax + (5*width), width)

    return bins # create_bins

#*********************************************
def gaussian(X: np.ndarray,
             p: np.ndarray) -> np.ndarray:
    '''
    Gaussian function for line fitting
    p[0]=norm
    p[1]=mean
    p[2]=sigma
    '''
    norm, mu, sig = p
    return (norm*(np.exp(-((X-mu)*(X-mu))/(2.0*sig*sig)))) # gaussian

#*********************************************
def skew_gaussian(X: np.ndarray,
                  p: np.ndarray) -> np.ndarray:
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
def residuals_skew_gaussian(p: np.ndarray,
                            Y: np.ndarray,
                            X: np.ndarray) -> np.ndarray:
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
def fit_gaussian(x: np.ma.MaskedArray, y: np.ma.MaskedArray,
                 norm: float,
                 mu: float = utils.MDI,
                 sig: float = utils.MDI,
                 skew: float = utils.MDI) -> np.ndarray:
    '''
    Fit a gaussian to the data provided
    Inputs:
      x - x-data
      y - y-data
      norm - norm
    Outputs:
      fit - array of [norm,mu,sigma,(skew)]
    '''
    if mu == utils.MDI:
        mu = np.ma.mean(x)
    if sig == utils.MDI:
        sig = np.ma.std(x)

    if sig == 0:
        # calculation of spread hasn't worked for some reason
        sig = 3.*np.unique(np.diff(x))[0]

    if np.isnan(skew):
        # calculation of skew hasn't worked for some reason
        # (e.g. no variation in values, all==0)
        skew = 0

    # call the appropriate fitting function and routine
    if skew == utils.MDI:
        p0 = np.array([norm, mu, sig])
        result = least_squares(residuals_gaussian, p0, args=(y, x),
                               max_nfev=10000, verbose=0,
                               method="trf", jac="3-point")
    else:
        p0 = np.array([norm, mu, sig, skew])
        result = least_squares(residuals_skew_gaussian, p0, args=(y, x),
                               max_nfev=10000, verbose=0,
                               method="trf", jac="3-point")
    return result.x # fit_gaussian

#************************************************************************
def find_gap(hist: np.ndarray, bins: np.ndarray,
             threshold: float, gap_size: int,
             upwards: bool = True) -> float:
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
    assert len(hist)+1 == len(bins)

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
def reporting_accuracy(indata: np.ma.MaskedArray, winddir: bool = False,
                       plots: bool = False) -> float:
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
            # TODO: check this in more detail at some point
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


def update_dataframe(df: pd.DataFrame,
                     indata: np.ndarray,
                     locations: np.ndarray,
                     column_name: str  ) -> None:
    """Update dataframe from station values corrected during QC
    At the moment, wind direction during calm spells only

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame holding complete station data
    indata : np.ndarray
        Updated station data for single variable
    locations : np.ndarray
        Locations where data need updating
    column_name : str
        Column in DataFrame to update
    """

    df.loc[locations, column_name] = indata[locations]