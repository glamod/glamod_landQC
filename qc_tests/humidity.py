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
from qc_tests.pressure import plot_pressure_distribution

HIGH_FLAGGING_THRESHOLD = 0.4

# To account for greatest precisions we'd likely receive
#    As an initial attempt, using about half the worst precision.
#    But 0.1C at minimum
SUPERSAT_TOLERANCE = {0.1: 0.1,
                      0.5: 0.3,
                      1.0: 0.5}

# Dewpoint (or Twet) depression streak tolerance.
#   From HadISDH, Willett et al, 2013 [10.5194/cp-9-657-2013] section 4.1,
#           Tw uncertainty (1sigma) is 0.15C
#   From HadCRUT, Brohan et al, 2006 [10.1029/2005JD006548] section 2.3.1.1 &
#                 Folland et al, 2002, [10.1029/2001GL012877] p1 (2sigma=0.4C)
#           T uncertainty (1sigma) is 0.2C
DPD_STREAK_TOLERANCE = 0.35

MIN_RH_DIFF_SPREAD = 1
MIN_TWET_DIFF_SPREAD = 1
RH_THRESHOLD = 2  # x IQR difference offset
TWET_THRESHOLD = 2

# To help distinguish between wet-temperatures in config dictionary keys
DPD_DICT_NAME_LOOKUP = {"dew_point_temperature" : "Td",
                        "wet_bulb_temperature" : "Tw"}

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

    # equality within measurement tolerance, so get absolute quanity (magnitude)
    dpd = np.abs(temperatures.data - wet_temperatures.data)

    # find the DPD<Tolerance locations, and then see if there are streaks
    locs, = np.ma.nonzero(dpd < DPD_STREAK_TOLERANCE)

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
            config_dict["HUMIDITY"][f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}"] = threshold
        except KeyError:
            # ensuring that threshold is stored as a float, not an np.array.
            CD_dpd = {f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}" : float(threshold)}
            config_dict["HUMIDITY"] = CD_dpd

    else:
        # store high value so threshold never reached (MDI already negative)
        try:
            config_dict["HUMIDITY"][f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}"] = -utils.MDI
        except KeyError:
            CD_dpd = {f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}" : float(-utils.MDI)}
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


    for year in np.unique(station.years):
        for month in range(1, 13):
            month_locs, = np.nonzero(np.logical_and(station.years == year,
                                                    station.months == month,
                                                    wet_temperatures.data.mask == True))

            if month_locs.shape[0] == 0:
                # no data in any variable
                continue

            if (len(temperatures.data[month_locs].compressed()) < utils.DATA_COUNT_THRESHOLD) or\
                (len(wet_temperatures.data[month_locs].compressed()) < utils.DATA_COUNT_THRESHOLD):
                # no data in either of the two variables
                continue

            # use precision information to set tolerance

            temps_precision = qc_utils.reporting_accuracy(temperatures.data[month_locs])
            wet_temps_precision = qc_utils.reporting_accuracy(wet_temperatures.data[month_locs])

            sss, = np.ma.nonzero(wet_temperatures.data[month_locs] > \
                (temperatures.data[month_locs] + SUPERSAT_TOLERANCE[max(temps_precision, wet_temps_precision)]))

            flags[month_locs[sss]] = "m"

            # and whole month of Tw/dewpoints if month has a high proportion (of dewpoint obs)
            if (sss.shape[0]/month_locs.shape[0]) > HIGH_FLAGGING_THRESHOLD:
                flags[month_locs] = "m"

    # only flag the Tw/dewpoints
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
        th = config_dict["HUMIDITY"][f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}"]
        threshold = float(th)
    except KeyError:
        # no threshold set
        get_repeating_dpd_threshold(temperatures, wet_temperatures, config_dict,
                                    plots=plots, diagnostics=diagnostics)
        th = config_dict["HUMIDITY"][f"DPD-{DPD_DICT_NAME_LOOKUP[wet_temperatures.name]}"]
        threshold = float(th)

    # equality within measurement tolerance, so get absolute quanity (magnitude)
    dpd = np.abs(temperatures.data - wet_temperatures.data)

    # find the DPD<Tolerance locations, and then see if there are streaks
    locs, = np.ma.nonzero(dpd < DPD_STREAK_TOLERANCE)

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
def _calculate_e_v_wrt_water(temperature: np.ma.MaskedArray,
                             pressure: np.ma.MaskedArray) -> np.ma.MaskedArray:
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
def _calculate_e_v_wrt_ice(temperature: np.ma.MaskedArray,
                           pressure: np.ma.MaskedArray) -> np.ma.MaskedArray:
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
def _calculate_Tw_stull(e_v: np.ma.MaskedArray,
                        e_s: np.ma.MaskedArray,
                        temperature: np.ma.MaskedArray) -> np.ma.MaskedArray:
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

    return Tw # calculate_Tw_stull

#************************************************************************
def get_vapor_pressures(temperatures: np.ma.MaskedArray,
                        dewpoints: np.ma.MaskedArray,
                        station_pressure: np.ma.MaskedArray) -> tuple[np.ma.MaskedArray,
                                                                      np.ma.MaskedArray]:
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

    # adjust for ice-bulbs
    e_v[temperatures <= 0] = e_v_ice[temperatures <= 0]
    e_s[temperatures <= 0] = e_s_ice[temperatures <= 0]

    return e_v, e_s


def calculate_Tw(temperatures: np.ma.MaskedArray,
                dewpoints: np.ma.MaskedArray,
                station_pressure: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Calculate wet bulb using Stull's formula, with adjustment
    for ice-bulb if T<0

    Parameters
    ----------
    temperatures : np.ma.MaskedArray
        Dry-bulb temperatures (C)
    dewpoints : np.ma.MaskedArray
        Dew point temperatures (C)
    station_pressure : np.ma.MaskedArray
        Station pressure (hPa)

    Returns
    -------
    np.ma.MaskedArray
        Wet bulb temperatures (C)
    """

    # get vapour pressures
    e_v = _calculate_e_v_wrt_water(dewpoints, station_pressure)
    e_v_ice = _calculate_e_v_wrt_ice(dewpoints, station_pressure)

    # saturation vapour_pressures
    e_s = _calculate_e_v_wrt_water(temperatures, station_pressure)
    e_s_ice = _calculate_e_v_wrt_ice(temperatures, station_pressure)

    # get pseudo wet-bulb temperatures
    calc_Tw = _calculate_Tw_stull(e_v, e_s, temperatures)
    calc_Tw_ice = _calculate_Tw_stull(e_v_ice, e_s_ice, temperatures)

    Tw = calc_Tw.copy()
    # and set ice bulb
    Tw[temperatures <= 0] = calc_Tw_ice[temperatures <= 0]

    return Tw


def get_noaa_rh(temperatures: np.ma.MaskedArray,
                dewpoints: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """NOAA formula to calculate rh

    Parameters
    ----------
    temperatures : np.ma.MaskedArray
        Air temperature array
    dewpoints : np.ma.MaskedArray
        Dewpoint temperature array

    Returns
    -------
    np.ma.MaskedArray
        Relative Humidity array
    """

    return np.round((((112.0 - (0.1 *temperatures) + dewpoints) /
             (112.0 + (0.9 * temperatures)))**8) * 100.0, 0)


def to_fahrenheit(indata: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Convert Celsius to Fahrenheit

    Parameters
    ----------
    indata : np.ma.MaskedArray
        Array of Celsius data

    Returns
    -------
    np.ma.MaskedArray
        Array of Fahrenheit data
    """
    return (1.8 * indata) + 32.


def to_celsius(indata: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Convert Fahrenheit to Celsius

    Parameters
    ----------
    indata : np.ma.MaskedArray
        Array of Fahrenheit data

    Returns
    -------
    np.ma.MaskedArray
        Array of Celsius data
    """
    return (indata - 32) * (5./9.)


def to_inches_hg(indata: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Convert hPa to inches of mercury

    Parameters
    ----------
    indata : np.ma.MaskedArray
        Pressure in hPa

    Returns
    -------
    np.ma.MaskedArray
        Pressure in inchesHg
    """
    return indata * 0.02953


def get_noaa_twet(temperatures: np.ma.MaskedArray,
                  dewpoints: np.ma.MaskedArray,
                  station_pressure: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Calculate wet bulb temperature from formula supplied by NOAA

    Parameters
    ----------
    temperatures : np.ma.MaskedArray
        Dry-bulb temperatures (C)
    dewpoints : np.ma.MaskedArray
        Dew point temperatures (C)
    station_pressure : np.ma.MaskedArray
        Station pressure (hPa)

    Returns
    -------
    np.ma.MaskedArray
        Wet bulb temperatures (C), to 0.1-degree precision
    """

    # convert to Fahrenheit and inches of mercury
    temperatureF = np.round(to_fahrenheit(temperatures))
    dewpointF = np.round(to_fahrenheit(dewpoints))
    mercury_stnp = np.round(to_inches_hg(station_pressure), 2)

    # set up empty arryes
    wetbulbF = np.ma.zeros(temperatures.data.shape)
    wetbulbF.mask = np.ones(wetbulbF.shape)

    # constants for formula
    a = (temperatureF - dewpointF) * 0.1
    b = a - 1.0
    c = a**2

    # using above/below 0F as the threshold for the two different formulae
    below_zeroF, = np.nonzero(temperatureF < 0.)
    above_zeroF, = np.nonzero(temperatureF >= 0.)

    # do calculation (in Fahrenheit)
    if len(below_zeroF > 0):
        wetbulbF[below_zeroF] = (temperatureF[below_zeroF] -
                                 ((0.034 * a[below_zeroF]) - (0.006 * c[below_zeroF])) *
                                 ((0.6 * (temperatureF[below_zeroF] + dewpointF[below_zeroF])) -
                                  ((2.0 * mercury_stnp[below_zeroF]) + 108.0)))
    else:
        wetbulbF[above_zeroF] = (temperatureF[above_zeroF] -
                                 ((0.034 * a[above_zeroF]) - (0.00072 * a[above_zeroF] * b[above_zeroF])) *
                                 ((temperatureF[above_zeroF] + dewpointF[above_zeroF]) -
                                  (2.0 * mercury_stnp[above_zeroF]) + 108.0))

    # return in Celsius, rounded to 1 decimal place
    return np.round(to_celsius(wetbulbF), 1)


def _calculate_rh_differences_noaa(temperatures: np.ndarray,
                                   dewpoints: np.ndarray,
                                   obs_rh: np.ndarray) -> np.ndarray:
    """Calculated differences between RH in data files to that
    from NOAA formulae

    Parameters
    ----------
    temperatures : np.ndarray
        Dry bulb temperature array
    dewpoints : np.ndarray
        Dew point temperature array
    obs_rh : np.ndarray
        Observed relative humidity array

    Returns
    -------
    np.ndarray
        Differences array (observed - NOAA derived)
    """
    # use NOAA formula to get rh
    noaa_rh = get_noaa_rh(temperatures, dewpoints)

    # differences between calculated and observed
    diffs = obs_rh - noaa_rh

    return diffs


def _calculate_twet_differences_noaa(temperatures: np.ndarray,
                                     dewpoints: np.ndarray,
                                     stn_pressure: np.ndarray,
                                     obs_twet: np.ndarray) -> np.ndarray:
    """Calculated differences between Twet in data files to that
    from NOAA formulae

    Parameters
    ----------
    temperatures : np.ndarray
        Dry bulb temperature array
    dewpoints : np.ndarray
        Dew point temperature array
    stn_pressure : np.ndarray
        Station level pressure array
    obs_twet : np.ndarray
        Observed wet bulb temperature array

    Returns
    -------
    np.ndarray
        Differences array (observed - NOAA derived)
    """
    # use NOAA formula to get Twet
    noaa_twet = get_noaa_twet(temperatures, dewpoints, stn_pressure)

    # differences between calculated and observed
    diffs = obs_twet - noaa_twet

    return diffs


def _calculate_rh_differences_full(temperatures: np.ndarray,
                                   dewpoints: np.ndarray,
                                   stnp: np.ndarray,
                                   obs_rh: np.ndarray) -> np.ndarray:
    """Calculated differences between RH in data files to that
    from standard formulae

    Parameters
    ----------
    temperatures : np.ndarray
        Dry bulb temperature array
    dewpoints : np.ndarray
        Dew point temperature array
    stnp : np.ndarray
        Station level pressure array
    obs_rh : np.ndarray
        Observed relative humidity array

    Returns
    -------
    np.ndarray
        Differences array (observed - derived)
    """
    # get the vapor pressure and saturation v.p.
    e_v, e_s = get_vapor_pressures(temperatures, dewpoints, stnp)

    # calculate rh from T & Td, and differences to observed
    calc_rh = (e_v / e_s) * 100.
    diffs = obs_rh - calc_rh

    return diffs


def _calculate_twet_differences_full(temperatures: np.ndarray,
                                     dewpoints: np.ndarray,
                                     stnp: np.ndarray,
                                     obs_twet: np.ndarray) -> np.ndarray:
    """Calculated differences between Twet in data files to that
    from standard formulae

    Parameters
    ----------
    temperatures : np.ndarray
        Dry bulb temperature array
    dewpoints : np.ndarray
        Dew point temperature array
    stnp : np.ndarray
        Station level pressure array
    obs_twet : np.ndarray
        Observed wet bulb temperature array

    Returns
    -------
    np.ndarray
        Differences array (observed - derived)
    """
    # get the vapor pressure and saturation v.p.
    calc_twet = calculate_Tw(temperatures, dewpoints, stnp)
    diffs = obs_twet - calc_twet

    return diffs


def _identify_and_store_obs_diffs_spread(diffs: np.ndarray,
                                        varname: str,
                                        config_dict: dict,
                                        plots: bool,
                                        is_noaa: bool):
    """Determine the distribution of the differences between
    the observed and derived RH, and store the spread in the
    configuration dictionary, for suitable thresholds

    Parameters
    ----------
    diffs : np.ndarray
        Differences between observed and derived RH
    varname : str
        Name of the variable being checked
    config_dict : dict
        Configuration dictionary to read critical values
    plots : bool
        Plots distribution of the differences
    is_noaa : bool
        If True, check using the NOAA formulae for derived values
    """
    if varname == "relative_humidity":
        key_start = "RH"
        min_spread = MIN_RH_DIFF_SPREAD
        threshold = RH_THRESHOLD
        longname = "Relative Humidity"
        units = "%rh"
    elif varname == "wet_bulb_temperature":
        key_start = "TW"
        min_spread = MIN_TWET_DIFF_SPREAD
        threshold = TWET_THRESHOLD
        longname = "Wet Bulb Temperature"
        units = "C"

    key_name = f"{key_start}-FULL"
    if is_noaa:
        key_name = f"{key_start}-NOAA"

    # e.g. if check_derived_only, but none are
    if len(diffs) < utils.DATA_COUNT_THRESHOLD:
        logger.info(f"{longname} Consistency - insufficient data")
        try:
            config_dict["HUMIDITY"][key_name] = -utils.MDI
        except KeyError:
            CD_diffs = {key_name : -utils.MDI}
            config_dict["HUMIDITY"] = CD_diffs
        return

    # find locations where rh differences are > N x spread
    #    increase spread if too small
    spread = qc_utils.spread(diffs)
    if spread < min_spread:
        spread = min_spread

    if plots:
        plot_pressure_distribution(diffs, f"{longname} Differences",
                                   vmin=-threshold * spread,
                                   vmax=threshold * spread,
                                   units=units)

    try:
        config_dict["HUMIDITY"][key_name] = spread
    except KeyError:
        CD_diffs = {key_name : spread}
        config_dict["HUMIDITY"] = CD_diffs


def _apply_flags(diffs: np.ndarray,
                 spread: float,
                 met_var: utils.MeteorologicalVariable,
                 flags: np.ndarray,
                 check_derived_only: bool) -> None:
    """Find locations where differences exceed threshold
    and flag

    Parameters
    ----------
    diffs : np.ndarray
        Differences between observed and derived RH
    spread : float
        Spread of distribution of the differnces
    obs_rh : utils.MeteorologicalVariable
        Observed relative humidity object
    flags : np.ndarray
        Flags array to be updated
    check_derived_only : bool
        If True, check using the NOAA formulae for derived values
    """
    if met_var.name == "relative_humidity":
        threshold = RH_THRESHOLD
    elif met_var.name == "wet_bulb_temperature":
        threshold = TWET_THRESHOLD

    bad_locs, = np.nonzero(np.abs(diffs) > threshold * spread)

    if len(bad_locs) != 0 :
        if check_derived_only:
            derived_flags = flags[met_var.is_derived]
            derived_flags[bad_locs] = "m"
            flags[met_var.is_derived] = derived_flags
        else:
            flags[bad_locs] = "m"
        met_var.store_flags(utils.insert_flags(met_var.flags, flags))


def rh_consistency_check(station: utils.Station,
                        config_dict: dict,
                        full: bool,
                        plots: bool,
                        diagnostics: bool,
                        check_derived_only: bool=True) -> None:
    """Compare recorded rh against that calculated from other metrics
    using NOAA formulae [or alternative formulae - Future work]

    Parameters
    ----------
    station : utils.Station
        Station object
    config_dict : dict
        configuration dictionary to store critical values
    full : bool
        run a full update and recalculate thresholds
    plots : bool
        turn on plots
    diagnostics : bool
        turn on diagnostic output
    check_derived_only : bool
        If True, check using the NOAA formulae for derived values
    """

    # pull out the relative humidity information
    obs_rh = getattr(station, "relative_humidity")
    if len(obs_rh.data.compressed()) == 0:
        return

    flags = np.array(["" for i in range(obs_rh.data.shape[0])])

    # pull out the remaining variables
    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")

    # get differences between derived and observed
    if check_derived_only:
        diffs = _calculate_rh_differences_noaa(temperatures.data[obs_rh.is_derived],
                                               dewpoints.data[obs_rh.is_derived],
                                               obs_rh.data[obs_rh.is_derived])
    else:
        stnp = getattr(station, "station_level_pressure")
        diffs = _calculate_rh_differences_full(temperatures.data,
                                               dewpoints.data,
                                               stnp.data,
                                               obs_rh.data)

    # find and store the spread
    if full:
        _identify_and_store_obs_diffs_spread(diffs, obs_rh.name,
                                            config_dict,
                                            plots=plots,
                                            is_noaa=check_derived_only)

    # read from the configuration dictionary
    try:
        spread = float(config_dict["HUMIDITY"][f"RH-NOAA"])
    except KeyError:
        # in case running full but no threshold available
        _identify_and_store_obs_diffs_spread(diffs, obs_rh.name,
                                             config_dict,
                                             plots=plots,
                                             is_noaa=check_derived_only)
        spread = float(config_dict["HUMIDITY"][f"RH-NOAA"])

    # apply the spread to identify and flag the bad observations
    _apply_flags(diffs, spread, obs_rh, flags, check_derived_only)

    logger.info(f"Relative Humidity Consistency (Derived): {obs_rh.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")


def twet_consistency_check(station: utils.Station,
                           config_dict: dict,
                           full: bool,
                           plots: bool,
                           diagnostics: bool,
                           check_derived_only: bool=True) -> None:
    """Compare recorded twet against that calculated from other metrics
    using the NOAA formulae [or alternative formulae - Future work]

    Parameters
    ----------
    station : utils.Station
        Station object
    config_dict : dict
        configuration dictionary to store critical values
    full : bool
        run a full update and recalculate thresholds
    plots : bool
        turn on plots
    diagnostics : bool
        turn on diagnostic output
    check_derived_only : bool
        If True, check using the NOAA formulae for derived values
    """

    # pull out the wet bulb information
    obs_twet = getattr(station, "wet_bulb_temperature")
    if len(obs_twet.data.compressed()) == 0:
        return

    flags = np.array(["" for i in range(obs_twet.data.shape[0])])

    # pull out the remaining variables
    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")
    stnp = getattr(station, "station_level_pressure")

    if check_derived_only:
        # Compare against NOAA formulae when these have been used.
        # calculate twet from T & Td, and differences to observed
        diffs = _calculate_twet_differences_noaa(temperatures.data[obs_twet.is_derived],
                                                 dewpoints.data[obs_twet.is_derived],
                                                 stnp.data[obs_twet.is_derived],
                                                 obs_twet.data[obs_twet.is_derived])

    else:
        # use alternative calculation of Twet for comparison
        diffs = _calculate_twet_differences_full(temperatures.data,
                                                 dewpoints.data,
                                                 stnp.data,
                                                 obs_twet.data)

    # find and store the spread
    if full:
        _identify_and_store_obs_diffs_spread(diffs, obs_twet.name,
                                             config_dict,
                                             plots=plots,
                                             is_noaa=check_derived_only)

    # read from the configuration dictionary
    try:
        spread = float(config_dict["HUMIDITY"][f"TW-NOAA"])
    except KeyError:
        # in case running full but no threshold available
        _identify_and_store_obs_diffs_spread(diffs, obs_twet.name,
                                             config_dict,
                                             plots=plots,
                                             is_noaa=check_derived_only)
        spread = float(config_dict["HUMIDITY"][f"TW-NOAA"])

    _apply_flags(diffs, spread, obs_twet, flags, check_derived_only)

    logger.info(f"Wet Bulb Temperature Consistency: {obs_twet.name}")
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

    # consistency checks, for derived values
    #    use T, Td to check rh and Tw are consistent with NOAA calculations
    #    Just to make sure nothing has gone wrong with that derivation
    rh_consistency_check(station, config_dict, full=full, plots=plots,
                         diagnostics=diagnostics)
    twet_consistency_check(station, config_dict, full=full,
                           plots=plots, diagnostics=diagnostics)

    # For future work
    # Consistency checks against other calculation methods (e.g. NEWT)



    # hcc

