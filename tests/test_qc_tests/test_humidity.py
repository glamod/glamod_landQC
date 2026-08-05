"""
Contains tests for humidity.py
"""
import pandas as pd
import numpy as np
import datetime as dt
from unittest.mock import patch, Mock
import pytest

import humidity

import common
import utils

def _setup_station() -> utils.Station:

    # set up the data
    temps = np.ma.arange(10.)
    temps.mask = np.zeros(len(temps))
    dewps = np.ma.arange(10.)-1.
    dewps.mask = np.zeros(len(dewps))

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    dew_point_temperature = common.example_test_variable("dew_point_temperature", dewps)

    # make Station
    station = common.example_test_station(temperature)
    station.dew_point_temperature = dew_point_temperature

    # and build the times, simple hourly stuff
    datetimes = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(dewps))])
    station.times = datetimes
    station.years = np.array([t.year for t in datetimes])
    station.months = np.array([t.month for t in datetimes])

    return station


@pytest.mark.parametrize("config_dict", [{}, {"HUMIDITY" : {}}])
def test_get_repeating_dpd_threshold_short_record(config_dict):

    # set up the data
    temps = np.arange(1)
    dewps = np.arange(1)-1

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    dew_point_temperature = common.example_test_variable("dew_point_temperature", dewps)

    humidity.get_repeating_dpd_threshold(temperature, dew_point_temperature, config_dict)

    assert config_dict["HUMIDITY"]["DPD-Td"] == -utils.MDI


@pytest.mark.parametrize("config_dict", [{}, {"HUMIDITY" : {}}])
def test_get_repeating_dpd_threshold(config_dict):

    # set up the data
    temps = np.arange(75)
    dewps = np.arange(75) - 2.
    # use same array as in qc_utils test
    locs = np.array([0,
                     10, 11, 12, 13, 14, 15, # 1 streak length 6
                     20, 21, 22, 23, # 2 length 4
                     30, 31, 32, 33,
                     40, 41, 42, # 3 length 3
                     50, 51, 52,
                     60, 61, 62,
                     70])
    # create the DPD=0
    dewps[locs] = temps[locs]


    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    dew_point_temperature = common.example_test_variable("dew_point_temperature", dewps)

    humidity.get_repeating_dpd_threshold(temperature, dew_point_temperature, config_dict)

    assert config_dict["HUMIDITY"]["DPD-Td"] == 7.0


# NOT TESTING PLOTTING

@patch("utils.DATA_COUNT_THRESHOLD", 1)
def test_super_saturation_check() -> None:

    station = _setup_station()
    # manually trigger the super saturation
    station.dew_point_temperature.data[:3] = station.temperature.data[:3]+1

    expected = np.array(["m", "m", "m", "", "", "", "", "", "", ""])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


@patch("utils.DATA_COUNT_THRESHOLD", 1)
def test_super_saturation_check_under_tolerance() -> None:

    station = _setup_station()
    # increase the precision of the data to 0.5C
    station.dew_point_temperature.data /= 2
    station.temperature.data /= 2

    # manually trigger the super saturation for he first 3, but final three below tolerance
    station.dew_point_temperature.data[:3] = station.temperature.data[:3]+0.5
    station.dew_point_temperature.data[-3:] = station.temperature.data[-3:]+0.1

    expected = np.array(["m", "m", "m", "", "", "", "", "", "", ""])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


@patch("utils.DATA_COUNT_THRESHOLD", 1)
def test_super_saturation_check_w_mask() -> None:

    station = _setup_station()

    # mask the first 3 entries
    station.temperature.data = np.ma.masked_where(station.temperature.data <=3,
                                                  station.temperature.data)
    station.dew_point_temperature.data = np.ma.masked_where(station.temperature.data <= 3,
                                                            station.dew_point_temperature.data)
    station.dew_point_temperature.data[-3:] = station.temperature.data[-3:]+1
    expected = np.array(["", "", "", "", "", "", "", "m", "m", "m"])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


@patch("utils.DATA_COUNT_THRESHOLD", 1)
def test_super_saturation_check_proportion() -> None:

    station = _setup_station()

    # of the 10 element array set over 40% as Super Saturated
    station.dew_point_temperature.data[:5] = station.temperature.data[:5]+1

    expected = np.array(["m", "m", "m", "m", "m", "m", "m", "m", "m", "m"])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


@pytest.mark.parametrize("bias, flags", ([0, ["m", "m", "m", "m", "m", "m"]],  # identical, flagged
                                         [0.3, ["m", "m", "m", "m", "m", "m"]], # within DPD Tolerance, flagged
                                         [0.6, ["", "", "", "", "", ""]]))  # larger than tolerance, not flagged
def test_dew_point_depression_streak(bias: float,
                                     flags: list[str]) -> None:
    """Test the dew point depression streak check, with a streak length of 5"""
    # streaks of length 5
    config_dict = {"HUMIDITY" : {"DPD-Td" : 5}}

    # set up the data
    temps = np.arange(75)
    dewps = np.arange(75) - 2.
    # use same array as in utils unit test
    locs = np.array([0,
                     10, 11, 12, 13, 14, 15,  # this set should be flagged if differnence < tolerance
                     20, 21, 22, 23,  # <-  all of these are too short
                     30, 31, 32, 33,
                     40, 41, 42,
                     50, 51, 52,
                     60, 61, 62,
                     70])

    expected = np.array(["" for _ in range(75)])
    expected[10:16] = flags

    # create the DPD<0.35 data
    dewps[locs] = temps[locs] - bias

    temperature = common.example_test_variable("temperature", temps)
    dewpoint = common.example_test_variable("dew_point_temperature", dewps)

    times = np.array([dt.datetime(2024, 1, 1, 12, 0) +
                     (i * dt.timedelta(seconds=60*60))
                     for i in range(len(dewps))])

    humidity.dew_point_depression_streak(times, temperature, dewpoint, config_dict)

    np.testing.assert_array_equal(dewpoint.flags, expected)


def test_dew_point_depression_streak_dict() -> None:
    """Test storing of configuration dictionary"""
    config_dict = {"HUMIDITY" : {}}
    # set up the data
    temps = np.arange(75)
    dewps = np.arange(75) - 2.

    temperature = common.example_test_variable("temperature", temps)
    dewpoint = common.example_test_variable("dew_point_temperature", dewps)

    times = np.array([dt.datetime(2024, 1, 1, 12, 0) +
                     (i * dt.timedelta(seconds=60*60))
                     for i in range(len(dewps))])

    humidity.dew_point_depression_streak(times, temperature, dewpoint, config_dict)

    assert config_dict["HUMIDITY"]["DPD-Td"] == -utils.MDI


# def test_calculate_e_v_wrt_water() -> None:
# def test_calculate_e_v_wrt_ice() -> None:
# def test_calculate_Tw() -> None:
# def test_get_vapor_pressures() -> None:
# def test_get_noaa_rh() -> None:

@pytest.mark.parametrize("celsius, fahrenheit", ((0, 32),
                                                 (37.8, 100),
                                                 (100, 212)))
def test_to_fahrenheit(celsius: float,
                       fahrenheit: float) -> None:
    """Test implementation of fahrenheit conversion"""

    result = humidity.to_fahrenheit(celsius)

    assert np.isclose(result, fahrenheit, atol=0.1)


@pytest.mark.parametrize("celsius, fahrenheit", ((0, 32),
                                                 (37.8, 100),
                                                 (100, 212)))
def test_to_celsius(celsius: float,
                    fahrenheit: float) -> None:
    """Test implementation of celsius conversion"""

    result = humidity.to_celsius(fahrenheit)

    assert np.isclose(result, celsius, atol=0.1)


@pytest.mark.parametrize("hpa, inches", ((1000, 29.5301),
                                         (900, 26.5771),
                                         (1100, 32.4831)))
def test_to_inches_hg(hpa: float,
                      inches: float) -> None:
    """Test conversion of hPa to inches mercury"""
    # https://convertlive.com/u/convert/hectopascals/to/inches-of-mercury
    result = humidity.to_inches_hg(hpa)

    assert np.isclose(result, inches, atol=0.1)


def test_get_noaa_twet() -> None:
    """Test NOAA Twet calculation for some example values"""

    # using ACW00011647 as of R8.1 as source for test data
    # 1958-1-1 0000 & 0600, + 2026-02-12 2100

    result = humidity.get_noaa_twet(np.array([25.0, 24.4, 26.4]),
                                    np.array([19.4, 20.0, 21.8]),
                                    np.array([1014.2, 1014.2, 1013.6]))

    np.testing.assert_array_almost_equal(result,
                                         np.array([21.4, 21.5, 23.3]), decimal=1)


def test_calculate_rh_differences_noaa() -> None:
    """Test calculation of differences to NOAA formula"""
    result = humidity._calculate_rh_differences_noaa(np.array([25.0, 24.4, 26.4]),
                                                     np.array([19.4, 20.0, 21.8]),
                                                     np.array([71.0, 77.0, 76.0]))

    np.testing.assert_array_almost_equal(result,
                                         np.array([0, 0, 0]), decimal=1)


# def test_calculate_rh_differences_full() -> None:

def test_calculate_twet_differences_noaa() -> None:
    """Test calculations of differences to NOAA Twet formula"""

    # using ACW00011647 as of R8.1 as source for test data
    # 1958-1-1 0000 & 0600, + 2026-02-12 2100

    result = humidity._calculate_twet_differences_noaa(np.array([25.0, 24.4, 26.4]),
                                                       np.array([19.4, 20.0, 21.8]),
                                                       np.array([1014.2, 1014.2, 1013.6]),
                                                       np.array([21.4, 21.5, 23.3]))

    np.testing.assert_array_almost_equal(result,
                                         np.array([0, 0, 0]), decimal=1)


# def test_calculate_twet_differences_full() -> None:

def test_identify_and_store_obs_diffs_spread_little_data() -> None:
    """Test function stores empty values if too little data"""
    config_dict = {"HUMIDITY" : {}}
    diffs = np.array([1.0, 2.0])  # Less than DATA_COUNT_THRESHOLD

    humidity._identify_and_store_obs_diffs_spread(diffs, "relative_humidity",
                                                 config_dict,
                                                 plots=False, is_noaa=True)

    assert config_dict["HUMIDITY"]["RH-NOAA"] == -utils.MDI


@pytest.mark.parametrize("spread, stored", ([1.5, 1.5],
                                            [1.0, 1.0],
                                            [0.5, 1.0]))
@patch("utils.DATA_COUNT_THRESHOLD", 1)
@patch("humidity.qc_utils.spread")
def test_identify_and_store_obs_diffs_spread(spread_mock: Mock,
                                            spread: float,
                                            stored: float)-> None:
    """Test function stores mocked values, spoofing the data count threshold"""

    config_dict = {"HUMIDITY" : {}}
    diffs = np.array([1.0, 2.0])

    spread_mock.return_value = spread

    humidity._identify_and_store_obs_diffs_spread(diffs, "relative_humidity",
                                                 config_dict,
                                                 plots=False, is_noaa=True)

    assert config_dict["HUMIDITY"]["RH-NOAA"] == stored


def test_apply_flags_rh() -> None:
    """Test the correct locations have flags set"""

    # some sensible RHs, all derived
    rhs = np.arange(55, 100, 5)
    relhum = common.example_test_variable("relative_humidity", rhs)
    setattr(relhum, "is_derived", np.ones(rhs.shape[0], dtype=bool))

    # All match NOAA apart from 2nd entry
    diffs = np.zeros(rhs.shape[0])
    diffs[1] = 10  # this should be flagged
    flags = np.array(["" for _ in range(rhs.shape[0])])

    # and generate the expected flags
    humidity._apply_flags(diffs, 2, relhum, flags, True)
    expected = flags[:]
    expected[1] = "m"

    np.testing.assert_array_equal(relhum.flags,
                                  expected)


# def test_twet_consistency_check() -> None:


@pytest.mark.parametrize("full", [True, False])
@patch("humidity.twet_consistency_check")
@patch("humidity.rh_consistency_check")
@patch("humidity.get_repeating_dpd_threshold")
@patch("humidity.dew_point_depression_streak")
@patch("humidity.super_saturation_check")
def test_read_hcc(supersat_check_mock: Mock,
                  dpd_check_mock: Mock,
                  get_threshold_mock: Mock,
                  rh_consistency_mock: Mock,
                  twet_consistency_mock: Mock,
                  full: bool) -> None:

    station = _setup_station()

    # Do the call
    humidity.hcc(station, {}, full=full)

    # Mock to check calls occur as expected (dew T and wet T)
    assert supersat_check_mock.call_count == 2
    assert dpd_check_mock.call_count == 2

    if full:
        assert get_threshold_mock.call_count == 2

    rh_consistency_mock.assert_called_once()
    twet_consistency_mock.assert_called_once()
