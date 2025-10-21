"""
Humidity Cross Checks
=====================

1. Check and flag instances of super saturation
2. Check and flag instances of dew point depression
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils
import qc_tests.qc_utils as qc_utils

HIGH_FLAGGING_THRESHOLD = 0.4
TOLERANCE = 1.e-10


#************************************************************************
def get_repeating_dpd_threshold(temperatures: utils.MeteorologicalVariable,
                                dewpoints: utils.MeteorologicalVariable,
                                config_dict: dict,
                                plots: bool = False,
                                diagnostics: bool = False) -> None:
    """
    Use distribution to determine threshold values.  Then also store in config dictionary.

    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # identical equality
    dpd = temperatures.data - dewpoints.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.where(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        repeated_streak_lengths, _, _ = qc_utils.prepare_data_repeating_streak(locs, diff=1, plots=plots, diagnostics=diagnostics)

        # bin width is 1 as dealing with the index.
        # minimum bin value is 2 as this is the shortest streak possible
        threshold = qc_utils.get_critical_values(repeated_streak_lengths, binmin=2,
                                              binwidth=1.0, plots=plots,
                                              diagnostics=diagnostics,
                                              title="DPD streak length",
                                              xlabel="Repeating DPD length")

        # write out the thresholds...
        try:
            config_dict["HUMIDITY"]["DPD"] = threshold
        except KeyError:
            # ensuring that threshold is stored as a float, not an np.array.
            CD_dpd = {"DPD" : float(threshold)}
            config_dict["HUMIDITY"] = CD_dpd

    else:
        # store high value so threshold never reached (MDI already negative)
        try:
            config_dict["HUMIDITY"]["DPD"] = -utils.MDI
        except KeyError:
            CD_dpd = {"DPD" : float(-utils.MDI)}
            config_dict["HUMIDITY"] = CD_dpd

    return # repeating_dpd_threshold

#*********************************************
def plot_humidities(T: utils.MeteorologicalVariable,
                    D: utils.MeteorologicalVariable,
                    times: np.ndarray,
                    bad: int) -> None:
    '''
    Plot each observation of SSS or DPD against surrounding data

    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints
    :param array times: datetime array
    :param int bad: the location of SSS
    '''
    import matplotlib.pyplot as plt

    pad_start = bad - 24
    if pad_start < 0:
        pad_start = 0
    pad_end = bad + 24
    if pad_end > len(T.data):
        pad_end = len(T.data)

    # simple plot
    plt.clf()
    plt.plot(times[pad_start : pad_end], T.data[pad_start : pad_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start : pad_end], D.data[pad_start : pad_end], 'b-', marker=".", label=D.name.capitalize())
    plt.plot(times[bad], D.data[bad], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(T.units)
    plt.show()

    return # plot_humidities

#*********************************************
def plot_humidity_streak(times: np.ndarray,
                         T: utils.MeteorologicalVariable,
                         D: utils.MeteorologicalVariable,
                         streak_start: int, streak_end: int) -> None:
    '''
    Plot each streak against surrounding data

    :param array times: datetime array
    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints
    :param int streak_start: the location of the streak
    :param int streak_end: the end of the DPD streak

    :returns:
    '''
    import matplotlib.pyplot as plt

    pad_start = streak_start - 48
    if pad_start < 0:
        pad_start = 0
    pad_end = streak_end + 48
    if pad_end > len(T.data.compressed()):
        pad_end = len(T.data.compressed())

    # simple plot
    plt.clf()
    plt.plot(times[pad_start: pad_end], T.data.compressed()[pad_start: pad_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start: pad_end], D.data.compressed()[pad_start: pad_end], 'b-', marker=".", label=D.name.capitalize())
    plt.plot(times[streak_start: streak_end], T.data.compressed()[streak_start: streak_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[streak_start: streak_end], D.data.compressed()[streak_start: streak_end], 'b-', marker=".", label=D.name.capitalize())

    plt.ylabel(T.units)
    plt.show()

    return # plot_humidity_streak

#************************************************************************
def super_saturation_check(station: utils.Station,
                           temperatures: utils.MeteorologicalVariable,
                           dewpoints: utils.MeteorologicalVariable,
                           plots: bool = False, diagnostics: bool = False) -> None:
    """
    Flag locations where dewpoint is greater than air temperature

    :param Station station: Station Object for the station
    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    sss, = np.ma.where(dewpoints.data > (temperatures.data + TOLERANCE))

    flags[sss] = "h"

    # and whole month of dewpoints if month has a high proportion (of dewpoint obs)
    for year in np.unique(station.years):
        for month in range(1, 13):
            month_locs, = np.nonzero(np.logical_and(station.years == year,
                                                  station.months == month,
                                                  dewpoints.data.mask == True))
            if month_locs.shape[0] != 0:
                flagged, = np.nonzero(flags[month_locs] == "h")
                if (flagged.shape[0]/month_locs.shape[0]) > HIGH_FLAGGING_THRESHOLD:
                    flags[month_locs] = "h"

    # only flag the dewpoints
    dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)

    # diagnostic plots
    if plots:
        for bad in sss:
            plot_humidities(temperatures, dewpoints, station.times, bad)

    logger.info(f"Supersaturation {dewpoints.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.nonzero(flags != '')[0])}")

    return # super_saturation_check

#************************************************************************
def dew_point_depression_streak(times: np.ndarray,
                                temperatures: utils.MeteorologicalVariable,
                                dewpoints: utils.MeteorologicalVariable,
                                config_dict: dict,
                                plots: bool = False,
                                diagnostics: bool = False) -> None:
    """
    Flag locations where dewpoint equals air temperature

    :param array times: datetime array
    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    # retrieve the threshold and store in another dictionary
    try:
        th = config_dict["HUMIDITY"]["DPD"]
        threshold = float(th)
    except KeyError:
        # no threshold set
        get_repeating_dpd_threshold(temperatures, dewpoints, config_dict, plots=plots, diagnostics=diagnostics)
        th = config_dict["HUMIDITY"]["DPD"]
        threshold = float(th)


    dpd = temperatures.data - dewpoints.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.where(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        repeated_streak_lengths, grouped_diffs, streaks = qc_utils.prepare_data_repeating_streak(locs, diff=1, plots=plots, diagnostics=diagnostics)

        # above threshold
        bad, = np.nonzero(repeated_streak_lengths >= threshold)

        # flag identified streaks
        for streak in bad:
            start = int(np.sum(grouped_diffs[:streaks[streak], 1]))
            end = start + int(grouped_diffs[streaks[streak], 1]) + 1
            flags[locs[start : end]] = "h"

            if plots:
                plot_humidity_streak(times, temperatures, dewpoints, start, end)

        # only flag the dewpoints
        dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)

    logger.info(f"Dewpoint Depression {dewpoints.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.nonzero(flags != '')[0])}")

    return # dew_point_depression_streak

#************************************************************************
def hcc(station: utils.Station, config_dict: dict,
        full: bool = False, plots: bool = False,
        diagnostics:bool = False) -> None:
    """
    Extract the variables and pass to the Humidity Cross Checks

    :param Station station: Station Object for the station
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")

    # Super Saturation
    super_saturation_check(station, temperatures, dewpoints, plots=plots, diagnostics=diagnostics)

    # Dew Point Depression
    #    Note, won't have cloud-base or past-significant-weather
    #    Note, currently don't have precipitation information
    if full:
        get_repeating_dpd_threshold(temperatures, dewpoints, config_dict, plots=plots, diagnostics=diagnostics)
    dew_point_depression_streak(station.times, temperatures, dewpoints, config_dict, plots=plots, diagnostics=diagnostics)

    # dew point cut-offs (HadISD) not run
    #  greater chance of removing good observations
    #  18 July 2019 RJHD

    return # hcc

