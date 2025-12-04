"""
Excess Variance Checks
======================

Checks for months with higher/lower variance than expected

"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_tests.qc_utils as qc_utils
import utils
#************************************************************************
STORM_THRESHOLD = 4
MIN_VARIANCES = 10
MIN_SPREAD = 1.5
SPREAD_THRESHOLD = 8.
MIN_VALUES = 30


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


def calculate_hourly_anomalies(hours: np.ndarray[int],
                               month_data: np.ndarray,
                               winsorize: bool=False) -> np.ma.MaskedArray:
    """Calculate anomaly values, using climatology for each hour in
    24, for this calendar month (i.e. all Januaries)

    Parameters
    ----------
    hours : np.ndarray
        Array of hours of the day for each observation in this month of the year
    month_data : np.ndarray
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


def normalise_hourly_anomalies(anomalies: np.ndarray) -> np.ndarray:
    """Normalise the anomalies by their spread (e.g. variance)

    Parameters
    ----------
    anomalies : np.ndarray
        Hourly anomalies to normalise by the spread

    Returns
    -------
    np.ndarray
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
                               month_locs: np.ndarray) -> np.ndarray:
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
    np.ndarray
        Array of variances for each year for this calendar month
    """

    # calculate the variance for each year in this single month.
    all_years = np.unique(stn_years)

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
                 winsorize: bool = True) -> np.ndarray:
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

    return # find_thresholds

#************************************************************************
def variance_check(obs_var: utils.MeteorologicalVariable, station: utils.Station, config_dict: dict,
                   plots: bool = False, diagnostics: bool = False, winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config file.

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
        month_locs, = np.where(station.months == month)

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
            continue

        bad_years, = np.where(np.abs(variances - average_variance) / variance_spread > SPREAD_THRESHOLD)

        # prepare wind and pressure data in case needed to check for storms
        if obs_var.name in ["station_level_pressure", "sea_level_pressure", "wind_speed"]:
            wind_monthly_data = station.wind_speed.data[month_locs]
            if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                pressure_monthly_data = obs_var.data[month_locs]
            else:
                pressure_monthly_data = station.sea_level_pressure.data[month_locs]

            if len(pressure_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
                    len(wind_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
                # need sufficient data to work with for storm check to work, else can't tell
                #    move on
                continue

            wind_average = qc_utils.average(wind_monthly_data)
            wind_spread = qc_utils.spread(wind_monthly_data)

            pressure_average = qc_utils.average(pressure_monthly_data)
            pressure_spread = qc_utils.spread(pressure_monthly_data)

        # go through each bad year for this month
        all_years = np.unique(station.years)
        for year in bad_years:

            # corresponding locations
            ym_locs, = np.where(np.logical_and(station.months == month, station.years == all_years[year]))

            # if pressure or wind speed, need to do some further checking before applying flags
            if obs_var.name in ["station_level_pressure", "sea_level_pressure", "wind_speed"]:

                # pull out the data
                wind_data = station.wind_speed.data[ym_locs]
                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    pressure_data = obs_var.data[ym_locs]
                else:
                    pressure_data = station.sea_level_pressure.data[ym_locs]


                # need sufficient data to work with for storm check to work, else can't tell
                if len(pressure_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
                        len(wind_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
                    # move on
                    continue

                # find locations of high wind speeds and low pressures, cross match
                high_winds, = np.ma.where((wind_data - wind_average)/wind_spread > STORM_THRESHOLD)
                low_pressures, = np.ma.where((pressure_average - pressure_data)/pressure_spread > STORM_THRESHOLD)

                match = np.in1d(high_winds, low_pressures)

                couldbe_storm = False
                if len(match) > 0:
                    # this could be a storm, either at tropical station (relatively constant pressure)
                    # or out of season in mid-latitudes.
                    couldbe_storm = True

                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    diffs = np.ma.diff(pressure_data)
                elif obs_var.name == "wind_speed":
                    diffs = np.ma.diff(wind_data)

                # count up the largest number of sequential negative and positive differences
                negs, poss = 0, 0
                biggest_neg, biggest_pos = 0, 0

                for diff in diffs:

                    if diff > 0:
                        if negs > biggest_neg: biggest_neg = negs
                        negs = 0
                        poss += 1
                    else:
                        if poss > biggest_pos: biggest_pos = poss
                        poss = 0
                        negs += 1

                if (biggest_neg < 10) and (biggest_pos < 10) and not couldbe_storm:
                    # insufficient to identify as a storm (HadISD values)
                    # leave flags set
                    pass
                else:
                    # could be a storm, so better to leave this month unflagged
                    # zero length array to flag
                    ym_locs = np.ma.array([])


            # copy over the flags, if any
            if len(ym_locs) != 0:
                # and set the flags
                flags[ym_locs] = "V"

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt

            scaled_variances = ((variances - average_variance) / variance_spread)
            bins = qc_utils.create_bins(scaled_variances, 0.25, obs_var.name)
            hist, bin_edges = np.histogram(scaled_variances, bins)

            plt.clf()
            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Months")
            plt.xlabel(f"Scaled {obs_var.name.capitalize()} Variances")
            plt.title(f"{station.id} - month {month}")

            plt.ylim([0.1, max(hist)*2])
            plt.axvline(SPREAD_THRESHOLD, c="r")
            plt.axvline(-SPREAD_THRESHOLD, c="r")

            bad_hist, dummy = np.histogram(scaled_variances[bad_years], bins)
            plt.step(bins[1:], bad_hist, color='r', where="pre")

            plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Variance {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # variance_check

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


    return # evc

