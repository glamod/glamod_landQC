"""
Humidity Cross Checks
=====================

1. Check and flag instances of super saturation
2. Check and flag instances of dew point depression
"""
#************************************************************************
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils
import qc_tests.qc_utils as qc_utils

HIGH_FLAGGING_THRESHOLD = 0.4
TOLERANCE = 1.e-10

RH_THRESHOLD = 2  # x IQR difference offset

#************************************************************************
def get_repeating_dpd_threshold(temperatures: utils.MeteorologicalVariable,
                                wet_temperatures: utils.MeteorologicalVariable,
                                config_dict: dict,
                                plots: bool = False,
                                diagnostics: bool = False) -> None:
    """
    Use distribution to determine threshold values.  Then also store in config dictionary.

    :param MetVar temperatures: temperatures object
    :param MetVar wet_temperatures: dewpoint or wet-bulb object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # identical equality
    dpd = temperatures.data - wet_temperatures.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.nonzero(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        (repeated_streak_lengths, _,
         _) = qc_utils.prepare_data_repeating_streak(locs, diff=1,
                                                     plots=plots, diagnostics=diagnostics)

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

    # repeating_dpd_threshold


#*********************************************
def plot_humidities(T: utils.MeteorologicalVariable,
                    D: utils.MeteorologicalVariable,
                    times: pd.Series,
                    bad: int) -> None:  # pragma: no cover
    '''
    Plot each observation of SSS or DPD against surrounding data

    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints/wetbulb
    :param Series times: datetime array
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
    plt.plot(times[pad_start : pad_end], T.data[pad_start : pad_end], 'k-',
             marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start : pad_end], D.data[pad_start : pad_end], 'b-',
             marker=".", label=D.name.capitalize())
    plt.plot(times[bad], D.data[bad], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(T.units)
    plt.show()

    # plot_humidities


#*********************************************
def plot_humidity_streak(times: pd.Series,
                         T: utils.MeteorologicalVariable,
                         D: utils.MeteorologicalVariable,
                         streak_locs: np.ndarray) -> None:  # pragma: no cover
    '''
    Plot each streak against surrounding data

    :param Series times: datetime array
    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints/wetbulb
    :param array streak_locs: locations of points in the DPD streak

    :returns:
    '''
    import matplotlib.pyplot as plt

    pad_start = streak_locs[0]- 48
    if pad_start < 0:
        pad_start = 0
    pad_end = streak_locs[-1] + 48
    if pad_end > len(T.data.compressed()):
        pad_end = len(T.data.compressed())

    # simple plot
    plt.clf()
    plt.plot(times[pad_start: pad_end], T.data[pad_start: pad_end],
             'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start: pad_end], D.data[pad_start: pad_end],
             'b-', marker=".", label=D.name.capitalize())
    plt.plot(times[streak_locs], T.data[streak_locs],
             'k-', marker="o", label=T.name.capitalize())
    plt.plot(times[streak_locs], D.data[streak_locs],
             'b-', marker="o", label=D.name.capitalize())

    plt.ylabel(T.units)
    plt.show()

    # plot_humidity_streak


#************************************************************************
def super_saturation_check(station: utils.Station,
                           temperatures: utils.MeteorologicalVariable,
                           wet_temperatures: utils.MeteorologicalVariable,
                           plots: bool = False, diagnostics: bool = False) -> None:
    """
    Flag locations where dewpoint or wet-bulb is greater than air temperature

    :param Station station: Station Object for the station
    :param MetVar temperatures: temperatures object
    :param MetVar wet_temperatures: dewpoints object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    sss, = np.ma.nonzero(wet_temperatures.data > (temperatures.data + TOLERANCE))

    flags[sss] = "m"

    # and whole month of dewpoints if month has a high proportion (of dewpoint obs)
    for year in np.unique(station.years):
        for month in range(1, 13):
            month_locs, = np.nonzero(np.logical_and(station.years == year,
                                                    station.months == month,
                                                    wet_temperatures.data.mask == True))
            if month_locs.shape[0] != 0:
                flagged, = np.nonzero(flags[month_locs] == "m")
                if (flagged.shape[0]/month_locs.shape[0]) > HIGH_FLAGGING_THRESHOLD:
                    flags[month_locs] = "m"

    # only flag the dewpoints
    wet_temperatures.store_flags(utils.insert_flags(wet_temperatures.flags, flags))

    # diagnostic plots
    if plots:
        for bad in sss:
            plot_humidities(temperatures, wet_temperatures, station.times, bad)

    logger.info(f"Supersaturation: {wet_temperatures.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")

    # super_saturation_check

#************************************************************************
def dew_point_depression_streak(times: pd.Series,
                                temperatures: utils.MeteorologicalVariable,
                                wet_temperatures: utils.MeteorologicalVariable,
                                config_dict: dict,
                                plots: bool = False,
                                diagnostics: bool = False) -> None:
    """
    Flag locations where dewpoint or wet-bulb equals air temperature

    :param Series times: datetime array
    :param MetVar temperatures: temperatures object
    :param MetVar wet_temperatures: dewpoints or wet-bulb temperatures object
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
        get_repeating_dpd_threshold(temperatures, wet_temperatures, config_dict,
                                    plots=plots, diagnostics=diagnostics)
        th = config_dict["HUMIDITY"]["DPD"]
        threshold = float(th)


    dpd = temperatures.data - wet_temperatures.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.nonzero(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        (repeated_streak_lengths, grouped_diffs,
         streaks) = qc_utils.prepare_data_repeating_streak(locs, diff=1,
                                                           plots=plots, diagnostics=diagnostics)

        # above threshold
        bad, = np.nonzero(repeated_streak_lengths >= threshold)

        # flag identified streaks
        for streak in bad:
            start = int(np.sum(grouped_diffs[:streaks[streak], 1]))
            end = start + int(grouped_diffs[streaks[streak], 1]) + 1
            flags[locs[start : end]] = "m"

            if plots:
                plot_humidity_streak(times, temperatures, wet_temperatures, locs[start: end])

        # only flag the dewpoints
        wet_temperatures.store_flags(utils.insert_flags(wet_temperatures.flags, flags))

    logger.info(f"Dewpoint Depression: {wet_temperatures.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")

    # dew_point_depression_streak


#************************************************************************
def _calculate_e_v_wrt_water(temperature, pressure):
    '''
    Calculate vapour pressure wrt water

    Buck, A. L.: New equations for computing vapor pressure and enhancement factor, J. Appl. Meteorol., 20, 1527?1532, 1981.

    :param array t: temperature (or dewpoint temperature for saturation e_v) (deg C)
    :param array P: station level pressure (hPa)

    :returns array e_v: vapour pressure (or saturation vapour
                        pressure if dewpoint temperature used) (hPa)
    '''

    f = 1 + (7.e-4) + ((3.46e-6) * pressure)

    e_v = 6.1121 * f * np.ma.exp(((18.729 - (temperature / 227.3)) *
                                  temperature) / (257.87 + temperature))

    return e_v # calculate_e_v_wrt_water

#************************************************************************
def _calculate_e_v_wrt_ice(temperature, pressure):
    '''
    Calculate vapour pressure wrt ice

    Buck, A. L.: New equations for computing vapor pressure and
    enhancement factor, J. Appl. Meteorol., 20, 1527?1532, 1981.

    :param array t: temperature (or dewpoint temperature for saturation e_v) (deg C)
    :param array P: station level pressure (hPa)

    :returns array e_v: vapour pressure (or saturation vapour
                        pressure if dewpoint temperature used) (hPa)
    '''

    f = 1 + (3.e-4) + ((4.18e-6) * pressure)

    e_v = 6.1115 * f * np.ma.exp(((23.036 - (temperature / 333.7)) *
                                  temperature) / (279.82 + temperature))

    return e_v # calculate_e_v_wrt_ice


#************************************************************************
def _calculate_Tw(e_v, e_s, temperature):
    '''
    Calculate the pseudo wetbulb temperature

    Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air Temperature, Journal of
    Applied Meteorology and Climatology, 50(11), 2267-2269. Retrieved Nov 10, 2022, from
    https://journals.ametsoc.org/view/journals/apme/50/11/jamc-d-11-0143.1.xml

    :param array e_v: vapour pressure (hPa)
    :param array e_s: saturation vapour pressure (hPa)
    :param array t: dry-bulb temperature (deg C)

    :returns array Tw: wetbulb temperature (deg C)
    '''

    rh = (e_v / e_s) * 100.

    Tw = (temperature * np.arctan(0.151977 * ((rh + 8.313659)**0.5))) +\
        np.arctan(temperature + rh) - np.arctan(rh - 1.676331) +\
        (0.00391838*((rh)**(3./2.)) * np.arctan(0.023101 * rh)) - 4.686035

    return Tw # calculate_Tw

#************************************************************************
def get_vapor_pressures(temperatures, dewpoints, station_pressure) -> tuple[np.ndarray,
                                                                            np.ndarray]:
    '''
    Calculate the vapour pressures and wet-bulb temperatures, adjusting
    for an ice- or water-bulb as appropriate from the calculated Tw

    :param array temperatures: temperature array
    :param array dewpoints: dewpoint temperature array
    :param array station_pressure: station pressure array

    :returns: e_v, e_s - vapour pressure, saturation vapour pressure
    '''

    # get vapour pressures
    e_v = _calculate_e_v_wrt_water(dewpoints, station_pressure)
    e_v_ice = _calculate_e_v_wrt_ice(dewpoints, station_pressure)

    # saturation vapour_pressures
    e_s = _calculate_e_v_wrt_water(temperatures, station_pressure)
    e_s_ice = _calculate_e_v_wrt_ice(temperatures, station_pressure)

    # get pseudo wet-bulb temperatures
    calc_Tw = _calculate_Tw(e_v, e_s, temperatures)
    calc_Tw_ice = _calculate_Tw(e_v_ice, e_s_ice, temperatures)

    # adjust for ice-bulbs
    e_v[calc_Tw <= 0] = e_v_ice[calc_Tw <= 0]
    e_s[calc_Tw <= 0] = e_s_ice[calc_Tw <= 0]

    return e_v, e_s


def rh_consistency_check(station: utils.Station,
                         plots: bool, diagnostics: bool) -> None:
    """Compare recorded rh against that calculated from other metrics

    Parameters
    ----------
    station : utils.Station
        Station object
    plots : bool
        turn on plots
    diagnostics : bool
        turn on diagnostic output
    """

    # pull out the relative humidity information
    obs_rh = getattr(station, "relative_humidity")
    if len(obs_rh.data.compressed()) == 0:
        return

    flags = np.array(["" for i in range(obs_rh.data.shape[0])])

    # pull out the remaining variables
    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")
    stnp = getattr(station, "station_level_pressure")

    # get the vapor pressure and saturation v.p.
    e_v, e_s = get_vapor_pressures(temperatures.data,
                                   dewpoints.data,
                                   stnp.data)

    # calculate rh from T & Td, and differences to observed
    calc_rh = (e_v / e_s) * 100.

    rh_diffs = obs_rh.data - calc_rh

    # find locations where rh differences are > N x spread
    spread = qc_utils.spread(rh_diffs)
    bad_locs, = np.nonzero(np.abs(rh_diffs) > RH_THRESHOLD * spread)

    if len(bad_locs) != 0 :
        flags[bad_locs] = "m"
        obs_rh.store_flags(utils.insert_flags(obs_rh.flags, flags))

    logger.info(f"Relative Humidity Consistency: {obs_rh.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")


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

    for var in ("dew_point_temperature", "wet_bulb_temperature"):
        comparison_temperatures = getattr(station, var)

        # Super Saturation check
        super_saturation_check(station, temperatures, comparison_temperatures,
                               plots=plots, diagnostics=diagnostics)

        # Dew Point Depression
        #    Note, won't have cloud-base or past-significant-weather
        #    Note, currently don't have precipitation information

        if full:
            get_repeating_dpd_threshold(temperatures, comparison_temperatures,
                                        config_dict, plots=plots, diagnostics=diagnostics)
        dew_point_depression_streak(station.times, temperatures, comparison_temperatures,
                                    config_dict, plots=plots, diagnostics=diagnostics)

    # dew point cut-offs (HadISD) not run
    #  greater chance of removing good observations
    #  18 July 2019 RJHD

    # relative humidity consistency
    #    use T, Td (and Tw?) to check rh is consistent

    rh_consistency_check(station, plots=plots, diagnostics=diagnostics)


    # hcc

