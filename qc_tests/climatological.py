"""
Climatological Outlier Check
============================

Check for observations which fall outside the climatologically expected range.
A low pass filter reduces the effect of long-term changes.

"""
#************************************************************************
import numpy as np
from scipy.stats import skew
import logging
logger = logging.getLogger(__name__)

import qc_tests.qc_utils as qc_utils
import utils
#************************************************************************

FREQUENCY_THRESHOLD = 0.1
GAP_SIZE = 2
BIN_WIDTH = 0.5


def plot_clims(hist: np.ndarray, bins: np.ndarray,
               xlabel: str, title: str,
               upper: float, lower: float,
               bad_hist: np.ndarray | None = None,
               curve: np.ndarray | None = None,) -> None:  # pragma: no cover
    """Plot the scaled climatologies and thresholds

    Parameters
    ----------
    hist : np.ndarray
        Histogram values to plot
    bins : np.ndarray
        Bin edges of the histogram
    xlabel : str
        X-axis label
    title : str
        Plot title
    upper : float
        Upper threshold
    lower : float
        Lower threshold

    Optional:
    bad_hist : np.ndarray
        Values to plot
    curve : np.ndarray
        Values to plot

    """
    import matplotlib.pyplot as plt
    plt.clf()
    plt.step(bins[1:], hist, color='k', where="pre")
    plt.yscale("log")

    plt.ylabel("Number of Observations")
    plt.xlabel(xlabel)
    plt.title(title)

    plt.ylim([0.1, max(hist)*2])
    plt.axvline(upper, c="r")
    plt.axvline(lower, c="r")

    # optionally - plot flagged months
    if bad_hist is not None:
        plt.step(bins[1:], bad_hist, color='r', where="pre")

    # optionally - plot result of curve fitting
    if curve is not None:
        bincentres = bins[1:] - (BIN_WIDTH/2)
        plt.plot(bincentres, curve)

    plt.show()

#************************************************************************
def get_weights(monthly_anoms: np.ndarray,
                monthly_subset: np.ndarray,
                filter_subset: np.ndarray) -> float:
    '''
    Get the weights for the low pass filter.

    :param array monthly_anoms: monthly anomalies
    :param array monthly_subset: which values to take
    :param array filter_subset: which values to take
    :returns:
        weights - float
    '''

    filterweights = np.array([1., 2., 3., 2., 1.])

    if np.sum(filterweights[filter_subset] * np.ceil(monthly_anoms[monthly_subset] - np.floor(monthly_anoms[monthly_subset]))) == 0:
        weights = 0.
    else:
        weights = np.sum(filterweights[filter_subset] * monthly_anoms[monthly_subset]) / \
            np.sum(filterweights[filter_subset] * np.ceil(monthly_anoms[monthly_subset] - np.floor(monthly_anoms[monthly_subset])))

    return weights # get_weights


def get_filter_ranges(year_index: int,
                      all_years: np.ndarray) -> tuple[np.ndarray,
                                                      np.ndarray]:
    """Get the ranges from which to calculate the filters.
    Need to handle the beginning and ends of the arrays

    Parameters
    ----------
    year_index : int
        Index of the year assessed in all_years
    all_years : np.ndarray
        Array of all the years present in the data

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        monthly_range - which data to choose, and
        filter_range - which filter values to apply
    """
    if year_index == 0:
        monthly_range = np.arange(0, 3)[:all_years.shape[0]]
        filter_range = np.arange(2, 5)[:all_years.shape[0]]
    elif year_index == 1:
        monthly_range = np.arange(0, 4)[:all_years.shape[0]]
        filter_range = np.arange(1, 5)[:all_years.shape[0]]
    elif year_index == all_years.shape[0] - 2:
        monthly_range = np.arange(-4, 0)[-all_years.shape[0]:]
        filter_range = np.arange(0, 4)[-all_years.shape[0]:]
    elif year_index == all_years.shape[0] - 1:
        monthly_range = np.arange(-3, 0)[-all_years.shape[0]:]
        filter_range = np.arange(0, 3)[-all_years.shape[0]:]
    else:
        monthly_range = np.arange(year_index-2, year_index+3)
        filter_range = np.arange(5)

    return (monthly_range, filter_range)

#************************************************************************
def low_pass_filter(normed_anomalies: np.ndarray, station: utils.Station,
                    annual_anoms: np.ndarray, month: int) -> np.ndarray:
    '''
    Run the low pass filter - get suitable ranges, get weights, and apply

    Using the annual average anomalies rather than anything tied to the month.

    :param array normed_anomalies: input normalised anomalies
    :param Station station: station object
    :param array annual_anoms: year average anomalies
    :param int month: month being processed

    :returns: normed_anomalies - with weights applied
    '''

    all_years = np.unique(station.years)
    for year in range(all_years.shape[0]):

        monthly_range, filter_range = get_filter_ranges(year,
                                                        all_years)

        if np.ma.sum(np.ma.abs(annual_anoms[monthly_range])) != 0:

            weights = get_weights(annual_anoms, monthly_range,
                                  filter_range)

            ymlocs, = np.where(np.logical_and(station.months == month,
                                              station.years == year))
            normed_anomalies[ymlocs] = normed_anomalies[ymlocs] - weights

    return normed_anomalies # low_pass_filter


def calculate_climatology(obs_var: utils.MeteorologicalVariable,
                          hmlocs: np.ndarray,
                          winsorize: bool=True) -> np.ma.MaskedArray:
    """Calculate the climatology for that hour of the day for that month

    Parameters
    ----------
    obs_var : utils.MeteorologicalVariable
        MetVar to work on
    hmlocs : np.ndarray
        locations for this hour and month
    winsorize : bool, optional
        Apply winsorization, by default True

    Returns
    -------
    np.ma.MaskedArray
        climatology, of length one, to retain mask information
    """

    hour_data = obs_var.data[hmlocs]

    if winsorize:
        if len(hour_data.compressed()) > 10:
            hour_data = qc_utils.winsorize(hour_data, 5)

    if len(hour_data) >= utils.DATA_COUNT_THRESHOLD:
        # TODO: check if should be  qc_utils.average
        clim = np.ma.mean(hour_data)
        return np.ma.array(clim, mask=False)

    else:
        return np.ma.array(0, mask=True)


def calculate_anomalies(station: utils.Station,
                        obs_var: utils.MeteorologicalVariable,
                        anomalies: np.ma.MaskedArray,
                        month: int,
                        winsorize: bool=True):
    """Calculate anomalies for each hour of the day
    for a single calendar month

    Parameters
    ----------
    station : utils.Station
        Station to work on (for month and hour information)
    obs_var : utils.MeteorologicalVariable
        MetVar to work on
    anomalies : np.ma.MaskedArray
        Container for anomalies
    month : int
        Which calendar month to work on
    winsorize : bool, optional
        Apply winsorization, by default True
    """

    hourly_clims = np.ma.zeros(24)
    hourly_clims.mask = np.ones(24)

    for hour in range(24):

        # calculate climatology
        hmlocs, = np.nonzero(np.logical_and(station.months == month,
                                            station.hours == hour))

        hourly_clims[hour] = calculate_climatology(obs_var, hmlocs,
                                                   winsorize=winsorize)

        # make anomalies - keeping the order
        anomalies[hmlocs] = obs_var.data[hmlocs] - hourly_clims[hour]


def normalise_anomalies(anomalies: np.ma.MaskedArray,
                        mlocs: np.ndarray) -> np.ma.MaskedArray:
    """Normalise anomalies by the spread

    Parameters
    ----------
    anomalies : np.ma.MaskedArray
        Array of the anomalies
    mlocs : np.ndarray
        Indices for values for this calendar month

    Returns
    -------
    np.ma.MaskedArray
        Normalised anomalies
    """
    # for the month, normalise anomalies by spread
    spread = qc_utils.spread(anomalies[mlocs])
    if spread < 1.5:
        spread = 1.5

    return anomalies[mlocs] / spread


def calculate_annual_anomalies(station: utils.Station,
                               obs_var: utils.MeteorologicalVariable) -> np.ma.MaskedArray:
    """Calculate anomalies over entire record to get an
    annual average.

    Parameters
    ----------
    station : utils.Station
        Station on which to work (for years information)
    obs_var : utils.MeteorologicalVariable
        Variable on which to find anomalies

    Returns
    -------
    np.ma.MaskedArray
        Annual anomalies
    """

    all_years = np.unique(station.years)  # returned sorted values
    monthly_anoms = np.ma.zeros(all_years.shape[0])
    for y, year in enumerate(all_years):

        ylocs, = np.nonzero(station.years == year)
        year_data = obs_var.data[ylocs]
        # No restrictions on data length, as monthly
        # routines will have already done that part
        monthly_anoms[y] = qc_utils.average(year_data)

    return monthly_anoms


#************************************************************************
def prepare_data(obs_var: utils.MeteorologicalVariable,
                 station: utils.Station,
                 month: int, diagnostics: bool=False,
                 winsorize: bool=True) -> np.ma.MaskedArray:
    """
    Prepare the data for the climatological check.
    Makes anomalies and applies low-pass filter

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: which month to run on
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%

    :returns: MaskedArray
    """

    anomalies = np.ma.zeros(obs_var.data.shape[0])
    anomalies.mask = np.ones(anomalies.shape[0])
    normed_anomalies = np.ma.copy(anomalies)

    mlocs, = np.where(station.months == month)

    nyears = len(np.unique(station.years[mlocs]))

    # need to have some data and in at least 5 years!
    if len(mlocs) >= utils.DATA_COUNT_THRESHOLD and nyears >= 5:

        anomalies.mask[mlocs] = False

        # modify anomalies array in place
        calculate_anomalies(station, obs_var, anomalies,
                            month, winsorize=winsorize)

        normed_anomalies.mask[mlocs] = False

        # if insufficient data at each hour, then no anomalies calculated
        if len(anomalies[mlocs].compressed()) >= utils.DATA_COUNT_THRESHOLD:

            # for the month, normalise anomalies by spread
            normed_anomalies[mlocs] = normalise_anomalies(anomalies, mlocs)

            # calculate values for the anomalies
            annual_anoms = calculate_annual_anomalies(station, obs_var)

            # apply low pass filter derived from annual anomalies
            lp_filtered_anomalies = low_pass_filter(normed_anomalies, station,
                                                    annual_anoms, month)

            return lp_filtered_anomalies  # prepare_data

        else:
            return anomalies  # prepare_data
    else:
        return anomalies  # prepare_data

#************************************************************************
def find_month_thresholds(obs_var: utils.MeteorologicalVariable,
                          station: utils.Station,
                          config_dict: dict, plots: bool = False,
                          diagnostics: bool = False,
                          winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    # get hourly climatology for each month
    for month in range(1, 13):

        normalised_anomalies = prepare_data(obs_var, station, month,
                                            diagnostics=diagnostics,
                                            winsorize=winsorize)

        if len(normalised_anomalies.compressed()) >= utils.DATA_COUNT_THRESHOLD:

            bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name)
            bincentres = bins[1:] - (BIN_WIDTH/2)
            hist, bin_edges = np.histogram(normalised_anomalies.compressed(), bins)

            # using skew-gaussian, and inflating spread to account for that in the fit
            # NOTE: for some of the recent exceptional extremes, skew gaussian
            #       may still not be sufficient, and need to think perhaps of an
            #       alternative distribution
            gaussian_fit = qc_utils.fit_gaussian(bincentres, hist, max(hist),
                                              mu=np.ma.median(normalised_anomalies),
                                              sig=1.5*qc_utils.spread(normalised_anomalies),
                                              skew=skew(normalised_anomalies.compressed())
            )

            if len(gaussian_fit) == 3:
                fitted_curve = qc_utils.gaussian(bincentres, gaussian_fit)
            elif len(gaussian_fit) == 4:
                fitted_curve = qc_utils.skew_gaussian(bincentres, gaussian_fit)

            # use bins and curve to find points where curve is < FREQUENCY_THRESHOLD
            #  round up or down to be fully encompassing
            try:
                lower_threshold = bins[:-1][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres < 0))[0]][-1]
            except:
                lower_threshold = bins[0]
            try:
                upper_threshold = bins[1:][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres > 0))[0]][0]
            except:
                upper_threshold = bins[-1]


            if plots:
                plot_clims(hist, bins,
                           f"Scaled {obs_var.name.capitalize()}",
                           f"{station.id} - month {month}",
                           upper_threshold, lower_threshold,
                           curve=fitted_curve)

            # Store values
            # add uthresh first, then lthresh
            try:
                config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"] = upper_threshold
            except KeyError:
                CD_uthresh = {f"{month}-uthresh" : upper_threshold}
                config_dict[f"CLIMATOLOGICAL-{obs_var.name}"] = CD_uthresh

            config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"] = lower_threshold

    # find_month_thresholds

#************************************************************************
def monthly_clim(obs_var: utils.MeteorologicalVariable, station: utils.Station, config_dict: dict,
                 logfile: str = "", plots: bool = False, diagnostics: bool = False, winsorize: bool = True):
    """
    Run through the variables and pass to the Distributional Gap Checks

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str configfile: dictionary for configuration settings
    :param str logfile: string for log file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    for month in range(1, 13):

        month_locs, = np.where(station.months == month)

        # note these are for the whole record, just this month is unmasked
        normalised_anomalies = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        if len(normalised_anomalies.compressed()) >= utils.DATA_COUNT_THRESHOLD:

            bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name)

            hist, _ = np.histogram(normalised_anomalies.compressed(), bins)

            try:
                upper_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"])
                lower_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"])
            except KeyError:
                find_month_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
                upper_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"])
                lower_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"])

            # now to find the gaps
            uppercount = len(np.where(normalised_anomalies > upper_threshold)[0])
            lowercount = len(np.where(normalised_anomalies < lower_threshold)[0])

            if uppercount > 0:
                gap_start = qc_utils.find_gap(hist, bins, upper_threshold, GAP_SIZE)

                if gap_start != 0:
                    bad_locs, = np.ma.where(normalised_anomalies > gap_start) # all years for one month

                    # normalised_anomalies are for the whole record, just this month is unmasked
                    flags[bad_locs] = "C"

            if lowercount > 0:
                gap_start = qc_utils.find_gap(hist, bins, lower_threshold, GAP_SIZE, upwards=False)

                if gap_start != 0:
                    bad_locs, = np.ma.where(normalised_anomalies < gap_start) # all years for one month

                    flags[bad_locs] = "C"

            # diagnostic plots
            if plots:
                bad_locs, = np.where(flags[month_locs] == "C")
                bad_hist, _ = np.histogram(normalised_anomalies[month_locs][bad_locs],
                                           bins)

                plot_clims(hist, bins,
                           f"Scaled {obs_var.name.capitalize()}",
                           f"{station.id} - month {month}",
                           upper_threshold, lower_threshold,
                           bad_hist=bad_hist)

    # append flags to object
    obs_var.store_flags(utils.insert_flags(obs_var.flags, flags))

    logger.info(f"Climatological {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # monthly_clim


#************************************************************************
def clim_outlier(station: utils.Station, var_list: list,
                 config_dict: dict, full: bool = False,
                 plots: bool = False,
                 diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Climatological Outlier Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # no point plotting twice!
        cplots = plots
        if plots and full:
            cplots = False

        if full:
            find_month_thresholds(obs_var, station, config_dict,
                                  plots=plots, diagnostics=diagnostics)
        monthly_clim(obs_var, station, config_dict,
                     plots=cplots, diagnostics=diagnostics)

    # clim_outlier

