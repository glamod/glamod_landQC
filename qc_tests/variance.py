"""
Excess Variance Checks
======================

Checks for months with higher/lower variance than expected

On a calendar month basis, get anomalies for each hour of the day
[For across all Januaries find difference of 15h00 obs from average of all January-15h00 obs].
Normalise hourly anomalies for this month by their spread.  Store
annual variance of these normalised anomalies (still per calendar month)
and compare with _their_ average and spread.  Find locations where average
exceeded by 8x spread. Check for locations where wind/pressure suggests
intense low pressure system has occurred (which would inflate the variance)
to retain these.

"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_tests.qc_utils as qc_utils
import utils
#************************************************************************
STORM_THRESHOLD = 4  # how much the wind/pressure needs to be greater/smaller than the average
MIN_VARIANCES = 10  # number of anomalies to enable calculation of spread for that month
MIN_SPREAD = 1.5  # minimum value for spread of anomalies
SPREAD_THRESHOLD = 8.  # how much the variance needs to exceed the average spread.
MIN_VALUES = 30  # minimum number of obs needed to get annual variance


def plot_variance_distribution(scaled_variances: np.ndarray,
                               bad_years: np.ndarray,
                               name: str, title: str) -> None:  # pragma: no cover
    """Plot the distribution of scaled annual variances
    for this month, highlighting those flagged"""

    import matplotlib.pyplot as plt


    bins = qc_utils.create_bins(scaled_variances, 0.25, name)
    hist, _ = np.histogram(scaled_variances, bins)

    plt.clf()
    plt.step(bins[1:], hist, color='k', where="pre")
    plt.yscale("log")

    plt.ylabel("Number of Months")
    plt.xlabel(f"Scaled {name.capitalize()} Variances")
    plt.title(title)

    plt.ylim([0.1, max(hist)*2])
    plt.axvline(SPREAD_THRESHOLD, c="r")
    plt.axvline(-SPREAD_THRESHOLD, c="r")

    bad_hist, _ = np.histogram(scaled_variances[bad_years], bins)
    plt.step(bins[1:], bad_hist, color='r', where="pre")

    plt.show()


def calculate_climatology(hour_data: np.ma.MaskedArray,
                          winsorize: bool=False) -> tuple[float, bool]:
    """Calculate the hourly climatology, with winsorizing if selected

    Parameters
    ----------
    hour_data : np.ma.MaskedArray
        Data for which to calculate the climatology
    winsorize : bool, optional
        Apply the winsorization, by default False

    Returns
    -------
    tuple[float, bool]
        mean and mask value.  If mean calculable, mask is False,
        else mean is 0 and mask is True
    """

    if len(hour_data.compressed()) >= utils.DATA_COUNT_THRESHOLD:
        if winsorize:
            hour_data = qc_utils.winsorize(hour_data, 5)

        return np.ma.mean(hour_data), False
    else:
        return 0, True


def calculate_hourly_anomalies(hours: np.ndarray,
                               month_data: np.ma.MaskedArray,
                               winsorize: bool=False) -> np.ma.MaskedArray:
    """Calculate anomaly values, using climatology for each hour in
    24, for this calendar month (i.e. all Januaries)

    Parameters
    ----------
    hours : np.ndarray
        Array of hours of the day for each observation in this month of the year
    month_data : np.ma.MaskedArray
        Data for all of this calendar month
    winsorize : bool, optional
        Apply the winsorization, by default False

    Returns
    -------
    np.ma.MaskedArray
        Anomalies for this calendar month using each hour of the day
    """
    assert isinstance(hours, np.ndarray)
    assert isinstance(month_data, np.ndarray)

    # set up the climatology and anomaly arrays
    hourly_clims = np.ma.zeros(24)
    hourly_clims.mask = np.ones(24)

    anomalies = np.ma.zeros(month_data.shape[0])
    anomalies.mask = np.ones(anomalies.shape[0])

    # spin through each hour of the day
    for hour in range(24):

        # calculate climatology
        hlocs, = np.where(hours == hour)

        (hourly_clims[hour],
         hourly_clims.mask[hour]) = calculate_climatology(month_data[hlocs],
                                                          winsorize=winsorize)

        # make anomalies - keeping the order
        anomalies[hlocs] = month_data[hlocs] - hourly_clims[hour]

    return anomalies


def normalise_hourly_anomalies(anomalies: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Normalise the anomalies by their spread (e.g. variance)

    Parameters
    ----------
    anomalies : np.ma.MaskedArray
        Hourly anomalies to normalise by the spread

    Returns
    -------
    np.ma.MaskedArray
        Normalised anomalies
    """

    if len(anomalies.compressed()) >= MIN_VARIANCES:
        # for the month, normalise anomalies by spread
        spread = qc_utils.spread(anomalies)
        if spread < MIN_SPREAD:
            spread = MIN_SPREAD
    else:
        spread = MIN_SPREAD

    return anomalies / spread


def calculate_yearly_variances(stn_years: np.ndarray,
                               anomalies: np.ma.MaskedArray,
                               month_locs: np.ndarray) -> np.ma.MaskedArray:
    """Calculate the variance for each year in the station
    for the calendar month selected in parent routine

    Parameters
    ----------
    stn_years : np.ndarray
        Year of observation for each timestamp
    anomalies : _type_
        Normalised anomaly at each timestamp
    month_locs : np.ndarray
        Locations in stn_years which correspond to this calendar month

    Returns
    -------
    np.ma.MaskedArray
        Array of variances for each year for this calendar month
    """

    # calculate the variance for each year in this single month.
    all_years = np.unique(stn_years)  #  unique returns sorted elements

    variances = np.ma.zeros(all_years.shape[0])
    variances.mask = np.ones(all_years.shape[0])
    for y, year in enumerate(all_years):

        ymlocs, = np.where(stn_years[month_locs] == year)
        this_year = anomalies[ymlocs]

        # HadISD used M.A.D.
        if this_year.compressed().shape[0] > MIN_VALUES:
            variances[y] = qc_utils.spread(this_year)

    return variances


#************************************************************************
def prepare_data(obs_var: utils.MeteorologicalVariable,
                 station: utils.Station, month:int,
                 diagnostics: bool = False,
                 winsorize: bool = True) -> np.ma.MaskedArray:
    """
    Calculate the monthly variances (each year for a given calendar month)

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: which month to run on
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """
    mlocs, = np.nonzero(station.months == month)
    month_data = obs_var.data[mlocs]

    # get hourly anomalies for this month, for each hour of the day
    anomalies = calculate_hourly_anomalies(station.hours[mlocs],
                                           month_data,
                                           winsorize=winsorize)

    # normalise by spread
    normed_anomalies = normalise_hourly_anomalies(anomalies)

    # find spread ("variance") of yearly anomalies
    variances = calculate_yearly_variances(station.years, normed_anomalies,
                                           mlocs)

    return variances # prepare_data


#************************************************************************
def find_thresholds(obs_var: utils.MeteorologicalVariable,
                    station: utils.Station, config_dict: dict,
                    plots: bool = False, diagnostics: bool = False,
                    winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config dictionary.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    # get hourly climatology for each month
    for month in range(1, 13):

        variances = prepare_data(obs_var, station, month,
                                 diagnostics=diagnostics, winsorize=winsorize)

        if len(variances.compressed()) >= MIN_VARIANCES:
            average_variance = qc_utils.average(variances)
            variance_spread = qc_utils.spread(variances)
        else:
            average_variance = utils.MDI
            variance_spread = utils.MDI

        try:
            config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-average"] =  average_variance
        except KeyError:
            CD_average = {f"{month}-average" : average_variance}
            config_dict[f"VARIANCE-{obs_var.name}"] = CD_average

        config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-spread"] = variance_spread

    # find_thresholds


def identify_bad_years(obs_var: utils.MeteorologicalVariable,
                       station: utils.Station, config_dict: dict,
                       month: int, plots: bool = False, diagnostics: bool = False,
                       winsorize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Use the settings in the config_dict (or calculate if not available)
    to return both the years for this calendar month where variances are suspect
    and the scaled variances.

    Parameters
    ----------
    obs_var : utils.MeteorologicalVariable
        Data structure for this variable
    station : utils.Station
        utils.Station object for the station
    config_dict : dict
        Dictionary holding the QC settings for this station
    month : int
        The calendar month to process
    plots : bool, optional
        Run plots, by default False
    diagnostics : bool, optional
        Diagnostic outout, by default False
    winsorize : bool, optional
        Apply winsorization, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of the indices of bad year, and the scaled variances
    """

    variances = prepare_data(obs_var, station, month,
                             diagnostics=diagnostics, winsorize=winsorize)

    try:
        average_variance = float(config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-average"])
        variance_spread = float(config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-spread"])
    except KeyError:
        find_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        average_variance = float(config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-average"])
        variance_spread = float(config_dict[f"VARIANCE-{obs_var.name}"][f"{month}-spread"])

    if average_variance == utils.MDI and variance_spread == utils.MDI:
        # couldn't be calculated, move on
        return (np.array([]), np.array([]))

    scaled_variances = ((variances - average_variance) / variance_spread)

    bad_years_locs, = np.where(np.abs(scaled_variances) > SPREAD_THRESHOLD)

    return bad_years_locs, scaled_variances


def read_wind_or_pressure(monthly_var: np.ma.MaskedArray) -> tuple[float, float]:
    """Return average and spread of data array supplied

    Parameters
    ----------
    monthly_var : np.ma.MaskedArray
        The data array for the variable

    Returns
    -------
    tuple[float, float]
        Values for average and spread
    """

    if len(monthly_var.compressed()) < utils.DATA_COUNT_THRESHOLD:
        return (-1., -1.)
    else:
        return (qc_utils.average(monthly_var),
                qc_utils.spread(monthly_var))


def high_wind_low_pressure_match(scaled_wind: np.ma.MaskedArray,
                                 scaled_pressure: np.ma.MaskedArray) -> bool:
    """Return locations where there are both high winds and low pressures,
    signifying that these could be low pressure storm system

    Parameters
    ----------
    scaled_wind : np.ma.MaskedArray
        Scaled wind anomalies ((data-average)/spread)
    scaled_pressure : _type_
        Scaled wind anomalies ((data-average)/spread)

    Returns
    -------
    bool
        True if there is at least one match of low pressure and high wind,
        else False
    """

    high_winds, = np.ma.where(scaled_wind > STORM_THRESHOLD)
    low_pressures, = np.ma.where(scaled_pressure > STORM_THRESHOLD)

    match = np.in1d(high_winds, low_pressures)

    couldbe_storm = False

    locs, = np.nonzero(match == True)
    if len(locs) > 0:
        # At least one observation of concurrent high wind
        #    and low pressure this month
        # This could be a storm, either at tropical station
        #    (relatively constant pressure)
        #    or out of season in mid-latitudes.
        couldbe_storm = True

    return couldbe_storm


def sequential_differences(diffs: np.ma.MaskedArray,
                           couldbe_storm: bool) -> bool:
    """Count the number of sequential differences in the quanitity.
    For a storm, expecting either a ramping up and down of wind, or down
    and up of pressure.  So want to calculate the number of sequential obs
    which step down/up.

    Parameters
    ----------
    diffs : np.ma.MaskedArray
        First difference array of quantity
    couldbe_storm : bool
        Indicator if high winds and low pressure values have occurred
        concurrently this month

    Returns
    -------
    bool
        True if sufficient indicators that storms may have occurred this
        month - and so set no flags.  Otherwise False
    """
    # count up the largest number of sequential negative and positive differences
    negs, poss = 0, 0
    biggest_neg_run, biggest_pos_run = 0, 0

    for diff in diffs:

        if diff > 0:
            if negs > biggest_neg_run:
                biggest_neg_run = negs
            # and reset counters
            negs = 0
            poss += 1
        else:
            if poss > biggest_pos_run:
                biggest_pos_run = poss
            # and reset counters
            poss = 0
            negs += 1

    # And do final check
    if negs > biggest_neg_run:
        biggest_neg_run = negs
    if poss > biggest_pos_run:
        biggest_pos_run = poss

    if (biggest_neg_run < 10) and (biggest_pos_run < 10) and not couldbe_storm:
        # insufficient to identify as a storm (HadISD values)
        # No runs of wind/pressure differences and no match of high wind/low pressure
        return False
    else:
        # could be a storm, so better to leave this month unflagged
        return True


def check_if_storm(station: utils.Station,
                   obs_var: utils.MeteorologicalVariable,
                   month_locs: np.ndarray,
                   ym_locs: np.ndarray) -> bool | np.ndarray:
    """Check if this year for this calendar month is potentially a storm
    which could result in above usual variance which should not be flagged

    Parameters
    ----------
    station : utils.Station
        Station which is being processed
    obs_var : utils.MeteorologicalVariable
        Observations being processed
    month_locs : np.ndarray
        Locations of this calendar month in the data array
    ym_locs : np.ndarray
        Locations of this year for this month in the data array

    Returns
    -------
    bool | np.ndarray
        Returns False if insufficient data to do the check.  Else returns
        locations (indices) which are to be flagged. This sequence of
        checks can remove all locations if they are likely enough to be a
        storm for this year-month
    """

    month_winds = station.wind_speed.data[month_locs]
    if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
        month_pressure = obs_var.data[month_locs]
    else:
        month_pressure = station.sea_level_pressure.data[month_locs]

    # this is knowingly duplicative, but as already in memory, logic is easier
    #    to do this step for each year.
    wind_average, wind_spread = read_wind_or_pressure(month_winds)
    pressure_average, pressure_spread = read_wind_or_pressure(month_pressure)

    # need enough data to work with
    if wind_average == wind_spread == -1 or\
        pressure_average == pressure_spread == -1:
        # set to be missing, so move on.
        return False

    # pull out the data for this year (and month)
    wind_data = month_winds[ym_locs]
    pressure_data = month_pressure[ym_locs]

    # need sufficient data to work with for storm check to work, else can't tell
    if len(pressure_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
            len(wind_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
        # move on
        return False

    scaled_wind = (wind_data - wind_average)/wind_spread
    scaled_pressure = (pressure_average - pressure_data)/pressure_spread

    couldbe_storm = high_wind_low_pressure_match(scaled_wind, scaled_pressure)

    if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
        diffs = np.ma.diff(pressure_data)
    elif obs_var.name == "wind_speed":
        diffs = np.ma.diff(wind_data)

    # count up the largest number of sequential negative and positive differences
    if sequential_differences(diffs, couldbe_storm):
        # this month has the potential to be a storm
        #   so do not set any flags
        return np.array([])

    return ym_locs


#************************************************************************
def variance_check(obs_var: utils.MeteorologicalVariable,
                   station: utils.Station, config_dict: dict,
                   plots: bool = False, diagnostics: bool = False,
                   winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.
    Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    # get hourly climatology for each month
    for month in range(1, 13):
        bad_years_locs, scaled_variances = identify_bad_years(obs_var, station,
                                                              config_dict, month,
                                                              plots=plots,
                                                              diagnostics=diagnostics,
                                                              winsorize=winsorize)

        # if no bad years, or variances calculable, move to next month
        if len(bad_years_locs) == 0 and len(scaled_variances) == 0:
            continue

        month_locs, = np.where(station.months == month)
        # go through each bad year for this month
        all_years = np.unique(station.years)  #  unique returns sorted elements

        for year in bad_years_locs:

            # corresponding locations
            ym_locs, = np.where(station.years[month_locs] == all_years[year])

            # if pressure or wind speed, need to do some
            # # further checking before applying flags
            if obs_var.name in ["station_level_pressure",
                                "sea_level_pressure",
                                "wind_speed"]:

                storm_check = check_if_storm(station, obs_var,
                                             month_locs, ym_locs)

                if storm_check is False:
                    # Checking explicitly for boolean False
                    # returned False, insufficient data to check
                    #    or not matching storm characteristics
                    pass
                else:
                    # Returned array, so overwrite flag locations
                    #    which may now be empty
                    ym_locs = storm_check

            # copy over the flags, if any
            if len(ym_locs) != 0:
                # and set the flags
                flags[month_locs[ym_locs]] = "V"

        # diagnostic plots
        if plots:
            plot_variance_distribution(scaled_variances,
                                       bad_years_locs,
                                       obs_var.name,
                                       f"{station.id} - month {month}")

    # append flags to object
    obs_var.store_flags(utils.insert_flags(obs_var.flags, flags))

    logger.info(f"Variance {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")

    # variance_check


#************************************************************************
def evc(station: utils.Station, var_list: list,
        config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Excess Variance Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        if full:
            find_thresholds(obs_var, station, config_dict,
                            plots=plots, diagnostics=diagnostics)
        variance_check(obs_var, station, config_dict,
                       plots=plots, diagnostics=diagnostics)


    # evc

