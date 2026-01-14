"""
Distributional Gap Checks - All Observations
============================================

Check distribution of all observations and look for distinct populations
"""
#************************************************************************
import numpy as np
import pandas as pd
from scipy.stats import skew
import logging
logger = logging.getLogger(__name__)

import qc_tests.qc_utils as qc_utils
import qc_tests.distribution_monthly as dist_monthly
import utils
#************************************************************************
STORM_THRESHOLD = 5

VALID_MONTHS = 5
MIN_OBS = 28*2 # two obs per day for the month (minimum subdaily)
SPREAD_LIMIT = 2 # two IQR/MAD/STD
BIN_WIDTH = 1.0
LARGE_LIMIT = 5
GAP_SIZE = 2
FREQUENCY_THRESHOLD = 0.1

#************************************************************************
def find_monthly_scaling(all_month_data: np.ma.MaskedArray,
                         config_dict: dict,
                         name: str,
                         month: int) -> tuple[float, float]:
    """Find climatology and spread to scale monthly data

    Parameters
    ----------
    all_month_data : np.ma.MaskedArray
        Data to characterise
    config_dict : dict
        Dictionary to store scaling
    name : str
        Name of variable
    month : int
        Month to run on

    Returns
    -------
    tuple[float, float]
        Returns climatology and spread
    """

    if len(all_month_data.compressed()) >= utils.DATA_COUNT_THRESHOLD:
        # have data, now to standardise
        climatology = qc_utils.average(all_month_data) # mean
        spread = qc_utils.spread(all_month_data) # IQR currently
    else:
        climatology = utils.MDI
        spread = utils.MDI

    # write out the scaling...
    try:
        config_dict[f"ADISTRIBUTION-{name}"][f"{month}-clim"] = climatology
    except KeyError:
        CD_clim = {f"{month}-clim" : climatology}
        config_dict[f"ADISTRIBUTION-{name}"] = CD_clim
    config_dict[f"ADISTRIBUTION-{name}"][f"{month}-spread"] = spread

    return climatology, spread


#************************************************************************
def prepare_all_data(obs_var: utils.MeteorologicalVariable,
                     station: utils.Station, month: int,
                     config_dict: dict, full: bool = False,
                     diagnostics: bool = False) -> np.ma.MaskedArray:
    """
    Extract data for the month, make & store or read average and spread.
    Use to calculate normalised anomalies.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param str config_dict: configuration dictionary to store critical values
    :param bool diagnostics: turn on diagnostic output

    :returns: np.ma.MaskedArray
    """

    month_locs, = np.nonzero(station.months == month)

    all_month_data = obs_var.data[month_locs]

    if full:
        climatology, spread = find_monthly_scaling(all_month_data, config_dict,
                                                   obs_var.name, month)

    else:
        try:
            # read if available
            climatology = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-clim"])
            spread = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-spread"])
        except KeyError:
            # will write to dictionary on recalculation
            climatology, spread = find_monthly_scaling(all_month_data, config_dict,
                                                       obs_var.name, month)

    if climatology == utils.MDI and spread == utils.MDI:
        # these weren't calculable, move on
        return np.ma.array([utils.MDI])
    elif spread == 0:
        # all observations have the same value (so all anomalies should be zero)
        return (all_month_data - climatology)  # prepare_all_data
    else:
        return (all_month_data - climatology)/spread  # prepare_all_data


#************************************************************************
def plot_thresholds(bins, hist, xlabel, title, upper_threshold, lower_threshold,
                    bincentres: np.ndarray | None = None,
                    fitted_curve: np.ndarray | None = None ,
                    bad_hist: np.ndarray | None = None) -> None:  # pragma: no cover
    """Plot the histogram and the threshold/flagged values"""

    import matplotlib.pyplot as plt
    plt.clf()
    plt.step(bins[1:], hist, color='k', where="pre")
    plt.yscale("log")

    plt.ylabel("Number of Observations")
    plt.xlabel(xlabel)
    plt.title(title)

    if bincentres is not None and fitted_curve is not None:
        plt.plot(bincentres, fitted_curve)
    plt.ylim([0.1, max(hist)*2])
    plt.axvline(upper_threshold, c="r")
    plt.axvline(lower_threshold, c="r")

    if bad_hist is not None:
        plt.step(bins[1:], bad_hist, color='r', where="pre")

    plt.show()


#************************************************************************
def write_thresh_to_config_dict(config_dict: dict,
                                var_name: str,
                                month: int,
                                upper_threshold: float,
                                lower_threshold: float) -> None:
    """Write the config dictionary for the thresholds

    Parameters
    ----------
    config_dict : dict
        Dictionary to hold the values
    var_name : str
        Variable being assessed
    month : int
        Month being assessed
    upper_threshold : float
        Upper threshold for flagging to begin
    lower_threshold : float
        Lower threshold for flagging to begin
    """

    # add uthresh first, then lthresh
    try:
        config_dict[f"ADISTRIBUTION-{var_name}"][f"{month}-uthresh"] = upper_threshold
    except KeyError:
        CD_uthresh = {f"{month}-uthresh" : upper_threshold}
        config_dict[f"ADISTRIBUTION-{var_name}"] = CD_uthresh
    config_dict[f"ADISTRIBUTION-{var_name}"][f"{month}-lthresh"] = lower_threshold


#************************************************************************
def find_thresholds(obs_var: utils.MeteorologicalVariable,
                    station: utils.Station,
                    config_dict: dict,
                    full: bool = False,
                    plots: bool = False,
                    diagnostics: bool = False) -> None:
    """
    Extract data for month and find thresholds in distribution and store.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param dict config_dict: configuration file to store critical values
    :param bool full: run a full update (recalculate scaling)
    :param bool plots: output plots
    :param bool diagnostics: turn on diagnostic output
    """

    # spin through each month
    for month in range(1, 13):

        normalised_anomalies = prepare_all_data(obs_var, station, month,
                                                config_dict, full=full,
                                                diagnostics=diagnostics)

        if len(normalised_anomalies.compressed()) == 1 and\
            normalised_anomalies[0] == utils.MDI:
            # scaling not possible for this month
            write_thresh_to_config_dict(config_dict, obs_var.name, month,
                              utils.MDI, utils.MDI)
            continue
        elif len(np.unique(normalised_anomalies)) == 1:
            # all the same value, so won't be able to fit a histogram
            write_thresh_to_config_dict(config_dict, obs_var.name, month,
                              utils.MDI, utils.MDI)
            continue

        bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH,
                                    obs_var.name, anomalies=True)
        bincentres = bins[1:] - (BIN_WIDTH/2)
        hist, _ = np.histogram(normalised_anomalies, bins)

        gaussian_fit = qc_utils.fit_gaussian(bincentres, hist, 0.5*max(hist),
                                          mu=np.ma.median(normalised_anomalies),
                                          sig=1.5*qc_utils.spread(normalised_anomalies),
                                          skew=0.5*skew(normalised_anomalies.compressed()))

        fitted_curve = qc_utils.skew_gaussian(bincentres, gaussian_fit)

        if len(np.unique(fitted_curve)) == 1:
            # just a line of zeros perhaps
            #   (found on AFA00409906 station_level_pressure 20190913)
            upper_threshold = bins[-1]
            lower_threshold = bins[0]
        else:
            # use bins and curve to find points where curve is < FREQUENCY_THRESHOLD
            try:
                lower_threshold = bincentres[np.nonzero(
                    np.logical_and(fitted_curve < FREQUENCY_THRESHOLD,
                                bincentres < bins[np.argmax(fitted_curve)]))[0]][-1]
            except IndexError:
                lower_threshold = bins[0]

            try:
                upper_threshold = bincentres[np.nonzero(
                        np.logical_and(fitted_curve < FREQUENCY_THRESHOLD,
                                    bincentres > bins[np.argmax(fitted_curve)]))[0]][0]
            except IndexError:
                upper_threshold = bins[-1]

        # and store:
        write_thresh_to_config_dict(config_dict, obs_var.name, month,
                          upper_threshold, lower_threshold)

        if plots:
            plot_thresholds(bins, hist, f"Normalised {obs_var.name.capitalize()} anomalies",
                            f"Normalised {obs_var.name.capitalize()} anomalies",
                            upper_threshold, lower_threshold,
                            bincentres=bincentres, fitted_curve=fitted_curve)

    # find_thresholds


#************************************************************************
def expand_around_storms(storms: np.ndarray,
                         maximum: int,
                         pad: int=6) -> np.ndarray:
    """
    Pad storm signal by N=6 hours

    :param np.ndarray storms: the indices which need padding
    :param int maximum: max length of the array to which these indices refer
    :param int pad: by how much to pad

    :returns: updated np.ndarray
    """
    for _ in range(pad):

        if storms[0]-1 >= 0:
            storms = np.insert(storms, 0, storms[0]-1)
        if storms[-1]+1 < maximum:
            storms = np.append(storms, storms[-1]+1)

    return np.unique(storms) # expand_around_storms


def check_through_storms(storms: np.ndarray,
                         times: pd.Series) -> np.ndarray:
    """Want to pad locations where pressure and wind speed suggest
    a low pressure system, where observations should be retained.
    Split into contiguous runs of when this is the case and pad either
    side.  Use the average reporting frequency for the station.

    Parameters
    ----------
    storms : np.ndarray
        Indices where pressure and wind speed are sufficiently
        anomalous to suggest a low pressure system
    times : pd.Series
        Station time points

    Returns
    -------
    np.ndarray
        Locations which could be a storm (plus a little bit)
        for flags to be removed from
    """


    # more than one location in this year-month combination
    if len(storms) >= 2:
        # find where separation more than the usual obs separation
        storm_1diffs = np.ma.diff(times[storms])
        separations, = np.nonzero(storm_1diffs > np.ma.median(np.ma.diff(times)))

        if len(separations) != 0:
            # multiple storm signals
            # Get the the first one, and expand
            final_storm_locs = expand_around_storms(
                storms[0: separations[0] + 1],
                len(times))

            # then do the rest in a loop
            for j in range(len(separations)):
                if j+1 == len(separations):
                    # final one
                    this_storm = expand_around_storms(
                        storms[separations[j]+1: ],
                        len(times))
                else:
                    this_storm = expand_around_storms(
                        storms[separations[j]+1: separations[j+1]+1],
                        len(times))

                final_storm_locs = np.append(final_storm_locs,
                                             this_storm)

        else:
            # locations separated at same interval as data
            final_storm_locs = expand_around_storms(storms, len(times))

    # single observation location identified
    elif len(storms) != 0:
        # expand around the storm signal (rather than
        #  just unflagging what could be the peak and
        #  leaving the entry/exit flagged)
        final_storm_locs = expand_around_storms(storms, len(times))

    return final_storm_locs


def average_and_spread(obs_var: utils.MeteorologicalVariable,
                       station: utils.Station,
                       month: int) -> tuple[float, float]:
    """Return average and spread of monthly average data

    Parameters
    ----------
    obs_var : utils.MeteorologicalVariable
        Variable to process
    station : utils.Station
        Station to process
    month : int
        Calendar month to process

    Returns
    -------
    tuple[float, float]
        Average, Spread
    """

    monthly_data = dist_monthly.prepare_monthly_data(obs_var, station, month)

    average = qc_utils.average(monthly_data)
    spread = qc_utils.spread(monthly_data)

    return (average, spread)


#************************************************************************
def find_storms(station: utils.Station,
                obs_var: utils.MeteorologicalVariable,
                month: int,
                flags: np.ndarray) -> None:
    """Find locations that could be from storms (low pressure systems),
    using wind speed and pressure data, and remove set flags from these
    as these data should be retained

    Parameters
    ----------
    station : utils.Station
        Station being processed
    obs_var : utils.MeteorologicalVariable
        Variable being processed
    month : int
        Calendar month being processed
    flags : np.ndarray
        Flag array to check (contains flags ["d"], which may be removed)
    """

    # need sufficient data to work with for storm check to work, else can't tell
    if len(obs_var.data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
            len(station.wind_speed.data.compressed()) < utils.DATA_COUNT_THRESHOLD:
        # Only applicable to wind speed, as pressure (obvs_var) will have been
        #   picked up earlier.  But clearer this way.
        return

    # calculate the monthly averages, and spread
    wind_monthly_average, wind_monthly_spread = average_and_spread(station.wind_speed,
                                                                   station, month)
    pressure_monthly_average, pressure_monthly_spread = average_and_spread(obs_var,
                                                                           station, month)

    # already a single calendar month, so go through each year
    all_years = np.unique(station.years)
    for year in all_years:
        # For this calendar month, for this year
        this_year_locs = np.nonzero(np.logical_and(station.years == year,
                                                   station.months == month))

        if "d" not in flags[this_year_locs]:
            # skip to next year if you get the chance
            continue

        # get data for this calendar year/month
        wind_data = station.wind_speed.data[this_year_locs]
        pressure_data = obs_var.data[this_year_locs]

        # find where windier and lower pressure than normal
        storms, = np.ma.where(np.logical_and(
            (((wind_data - wind_monthly_average)/wind_monthly_spread) > STORM_THRESHOLD),
            (((pressure_monthly_average - pressure_data)/pressure_monthly_spread) > STORM_THRESHOLD)
            ))

        # potentially unset the flags
        if len(storms) > 0:
            # maybe more than one entry - check if separate events
            final_storm_locs = check_through_storms(storms, station.times)

            year_flags = flags[this_year_locs]
            year_flags[final_storm_locs] = ""
            flags[this_year_locs] = year_flags



#************************************************************************
def all_obs_gap(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                config_dict: dict, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Extract data for month and find secondary populations in distribution.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    for month in range(1, 13):

        # use stored values unless unavailable
        normalised_anomalies = prepare_all_data(obs_var, station, month,
                                                config_dict, full=False,
                                                diagnostics=diagnostics)

        if (len(normalised_anomalies.compressed()) == 1 and normalised_anomalies[0] == utils.MDI):
            # no data to work with for this month, move on.
            continue

        bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name, anomalies=True)
        hist, _ = np.histogram(normalised_anomalies, bins)

        try:
            upper_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"])
            lower_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"])
        except KeyError:
            # create, using stored values if available
            find_thresholds(obs_var, station, config_dict, full=False,
                            plots=plots, diagnostics=diagnostics)
            upper_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"])
            lower_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"])


        if upper_threshold == utils.MDI and lower_threshold == utils.MDI:
            # these weren't able to be calculated, move on
            continue
        elif len(np.unique(normalised_anomalies)) == 1:
            # all the same value, so won't be able to fit a histogram
            continue

        # now to find the gaps
        uppercount = np.count_nonzero(normalised_anomalies > upper_threshold)
        lowercount = np.count_nonzero(normalised_anomalies < lower_threshold)


        month_locs, = np.nonzero(station.months == month) # append should keep year order
        if uppercount > 0:
            gap_start = qc_utils.find_gap(hist, bins, upper_threshold, GAP_SIZE)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies > gap_start) # all years for one month

                month_flags = flags[month_locs]
                month_flags[bad_locs] = "d"
                flags[month_locs] = month_flags

        if lowercount > 0:
            gap_start = qc_utils.find_gap(hist, bins, lower_threshold, GAP_SIZE, upwards=False)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies < gap_start) # all years for one month

                month_flags = flags[month_locs]
                month_flags[bad_locs] = "d"

                # for pressure data, see if the flagged obs correspond with high winds
                # could be a storm signal
                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    find_storms(station, obs_var, month, flags)

                # having checked for storms now store final flags
                flags[month_locs] = month_flags


        # diagnostic plots
        if plots:
            bad_locs, = np.nonzero(flags[month_locs] == "d")
            bad_hist, _ = np.histogram(normalised_anomalies[bad_locs], bins)

            plot_thresholds(bins, hist, f"Normalised {obs_var.name.capitalize()} anomalies",
                            f"Normalised {obs_var.name.capitalize()} anomalies",
                            upper_threshold, lower_threshold,
                            bad_hist=bad_hist)

    # append flags to object
    obs_var.store_flags(utils.insert_flags(obs_var.flags, flags))


    logger.info(f"Distribution (all) {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")

    # all_obs_gap

#************************************************************************
def dgc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the complete Distributional Gap Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration information
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # no point plotting twice!
        gplots = plots
        if plots and full:
            gplots = False

        # all observations gap
        if full:
            find_thresholds(obs_var, station, config_dict, full=True,
                            plots=plots, diagnostics=diagnostics)
        all_obs_gap(obs_var, station, config_dict,
                    plots=gplots, diagnostics=diagnostics)


    # dgc

