"""
Contains tests for streaks.py
"""
import numpy as np
import datetime as dt
import pandas as pd
import pytest

from unittest.mock import patch, Mock
from typing import Optional

import streaks
import qc_utils

import common

EXCESS_STREAK_STARTS_LENGTHS = {100: 15,
                                200: 15,
                                300: 15,
                                400: 15,
                                500: 15,
                                600: 15,
                                700: 15,
                                800: 15,
                                900: 15,
                                1000: 15,
                                1100: 15,
                                1200: 15,
                                1300: 15,
                                1400: 15,
                                1500: 15,
                                1600: 15,
                                1700: 15,
                                1800: 15,
                                1900: 15,
                                }



def _make_station(data: np.array, name: str,
                  times: Optional[np.array] = None) -> qc_utils.Station:
    """
    Create an example station 

    :param array data: data array (presumed temperatures)
    :param str name: name for variable

    :returns: Station
    
    """
    obs_var = common.example_test_variable(name, data)
    station = common.example_test_station(obs_var, times)

    return station


def _make_simple_masked_data(stop: int, step: float) -> np.array:

    data = np.ma.arange(0, stop, step).astype(float)
    data.mask = np.zeros(data.shape[0])

    return data


def _make_repeating_value_station(name: str) -> qc_utils.Station:

    # set up data
    data = _make_simple_masked_data(200, 0.1)
    station = _make_station(data, name)
    obs_var = getattr(station, name)

    # make the streak, but no flags
    obs_var.data[50:70] = 5

    return station

@pytest.fixture()
def station_for_rsc_logic() -> qc_utils.Station:
    def _make_rsc_station() -> qc_utils.Station:
        return _make_station(_make_simple_masked_data(100, 1), "temperature")
    return _make_rsc_station()

# not testing plotting

@patch("qc_utils.reporting_accuracy")
def test_mask_calms(reporting_mock: Mock) -> None:

    # 0.1 m/s resolution
    reporting_mock.return_value = 0.1

    # set up data
    data = np.ma.arange(2).astype(float)
    data.mask = np.zeros(data.shape[0])
    station = _make_station(data, "wind_speed")
    this_var = getattr(station, "wind_speed")

    # call routine
    streaks.mask_calms(this_var)

    assert this_var.data.mask[0] == True
    assert this_var.data.mask[1] == False


def test_get_repeating_streak_threshold() -> None:

    # make some streaks (set start index and length)
    data = common.generate_streaky_data(_make_simple_masked_data(200, 0.1),
                                        common.REPEATED_STREAK_STARTS_LENGTHS)

    station = _make_station(data, "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}

    streaks.get_repeating_streak_threshold(this_var, config_dict)

    assert config_dict["STREAK-temperature"]["Straight"] == 8


def test_get_repeating_streak_threshold_no_data() -> None:

    # set up data
    station = _make_station(_make_simple_masked_data(1, 1), "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}

    streaks.get_repeating_streak_threshold(this_var, config_dict)

    assert config_dict["STREAK-temperature"]["Straight"] == qc_utils.MDI


@patch("streaks.utils.get_critical_values")
def test_get_excess_streak_threshold(critical_values_mock: Mock) -> None:

    expected_proportions = []
    all_data = np.array([])
    years = np.array([])
    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])

    # add a some years with no streaks
    for y in range(10):
        all_data = np.append(all_data, data)
        years = np.append(years, (2000*np.ones(data.shape[0]))+y)
        expected_proportions += [0]

    data = common.generate_streaky_data(data, EXCESS_STREAK_STARTS_LENGTHS)

    # Add some years with streaks
    for y in range(10, 15):
        all_data = np.append(all_data, data)
        years = np.append(years, (2000*np.ones(data.shape[0]))+y)
        expected_proportions += [np.sum([v for _, v in EXCESS_STREAK_STARTS_LENGTHS.items()])/data.shape[0]]

    station = _make_station(all_data, "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}
    critical_values_mock.return_value = 0.3

    streaks.get_excess_streak_threshold(this_var, years, config_dict, plots=True)

    # Test that propotions were calculated as expected
    np.testing.assert_array_equal(np.array(expected_proportions),
                                  critical_values_mock.call_args.args[0])


def test_repeating_value_nonwind() -> None:

    # set up data
    station = _make_repeating_value_station("temperature")
    this_var = getattr(station, "temperature")

    expected_flags = np.array(["" for _ in this_var.data])
    expected_flags[50:70] = "K"

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : 10}  # streaks of 10 or more identical values
    config_dict["STREAK-temperature"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_repeating_value_threshold_is_mdi() -> None:

    station = _make_repeating_value_station("temperature")
    this_var = getattr(station, "temperature")

    expected_flags = np.array(["" for _ in this_var.data])

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : qc_utils.MDI}
    config_dict["STREAK-temperature"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_repeating_value_short() -> None:

    # set up data
    station = _make_station(_make_simple_masked_data(1, 1), "temperature")
    this_var = getattr(station, "temperature")

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : 10}  # streaks of 10 or more identical values
    config_dict["STREAK-temperature"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, np.array([""]))


def test_repeating_value_wind() -> None:

    # set up data
    station = _make_repeating_value_station("wind_speed")
    this_var = getattr(station, "wind_speed")

    # streak of zeros [calms] at the end, which should be ignored
    this_var.data[160:180] = 0
    expected_flags = np.array(["" for _ in this_var.data])
    expected_flags[50:70] = "K"

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : 10}  # streaks of 10 or more identical values
    config_dict["STREAK-wind_speed"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_excess_repeating_value():
    # initialise arrays
    expected_flags = np.array([])
    all_data = np.ma.array([])
    years = np.array([])

    # set up data, including some masked data
    data = np.ma.arange(0, 2000, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])
    data.mask[10] = True
    flags = np.array(["" for _ in data])

    # add a year with no streaks
    all_data = np.ma.append(all_data, data)
    expected_flags = np.append(expected_flags, flags)
    years = np.append(years, 2000*np.ones(data.shape[0]))

    # will only find streaks over a particular length
    #  i.e. not long enough to be flagged themselves, but too many if all together 
    #       in a single year
    data = common.generate_streaky_data(data, EXCESS_STREAK_STARTS_LENGTHS)
    for start, length in EXCESS_STREAK_STARTS_LENGTHS.items():
        if not data.mask[start]:
            # only mark expected flag if data not masked
            flags[start: start+length] = "x"

    # Add a year with streaks
    all_data = np.ma.append(all_data, data)
    years = np.append(years, 2001*np.ones(data.shape[0]))
    expected_flags = np.append(expected_flags, flags)

    # create the station and ObsVar
    times = pd.to_datetime(pd.DataFrame([dt.datetime(yr, 1, 1, 12, 0) + dt.timedelta(hours=y)\
                              for y, yr in enumerate(years.astype(int))])[0])

    station = _make_station(all_data, "temperature", times=times)
    this_var = getattr(station, "temperature")

    # set up dictionary, with a threshold to trigger the test
    config_dict = {}
    CD_excess = {"Excess" : 0.02}  
    config_dict["STREAK-temperature"] = CD_excess

    streaks.excess_repeating_value(this_var, station.times, config_dict)

    # assert flags are as expected
    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_repeating_day() -> None:

    # set up the data (mask one day too)
    indata = np.ma.arange(24*8) # 8 days
    indata.mask = np.zeros(indata.shape[0])
    indata.mask [-23:] = True
    expected_flags = np.array(["" for _ in indata])

    # add a streak of 3 repeated days (days 2-5)
    indata[72:96] = indata[48:72]
    indata[96:120] = indata[48:72]
    expected_flags[48:120] = "y"

    obs_var = common.example_test_variable("temperature", indata)
    station = common.example_test_station(obs_var)

    # set up config_dict
    config_dict = {}
    CD_dayrepeat = {"DayRepeat" : 2}
    config_dict[f"STREAK-{obs_var.name}"] = CD_dayrepeat    
    
    streaks.repeating_day(obs_var, station, config_dict, determine_threshold=False)

    np.testing.assert_array_equal(expected_flags, obs_var.flags)


# def test_hourly_repeat():

@pytest.mark.parametrize("full", [True, False])
@patch("streaks.get_repeating_streak_threshold")
@patch("streaks.repeating_value")
def test_rsc_repeating(repeating_value_mock: Mock,
                       get_threshold_mock: Mock,
                       full: bool,
                       station_for_rsc_logic: qc_utils.Station) -> None:

    # Do the call
    streaks.rsc(station_for_rsc_logic, ["temperature"], {}, full=full)

    # Mock to check call occurs as expected with right return
    repeating_value_mock.assert_called_once()
    if full:
        get_threshold_mock.assert_called_once()
    else:
        get_threshold_mock.assert_not_called()


@pytest.mark.parametrize("full", [True, False])
@patch("streaks.get_excess_streak_threshold")
@patch("streaks.excess_repeating_value")
def test_rsc_excess(excess_repeating_value_mock: Mock,
                    get_threshold_mock: Mock,
                    full: bool,
                    station_for_rsc_logic: qc_utils.Station) -> None:

    # Do the call
    streaks.rsc(station_for_rsc_logic, ["temperature"], {}, full=full)

    # Mock to check call occurs as expected with right return
    excess_repeating_value_mock.assert_called_once()
    if full:
        get_threshold_mock.assert_called_once()
    else:
        get_threshold_mock.assert_not_called()
    

@pytest.mark.parametrize("full", [True, False])
@patch("streaks.repeating_day")
def test_rsc_day(repeating_day: Mock,
                 full: bool,
                 station_for_rsc_logic: qc_utils.Station) -> None:

    streaks.rsc(station_for_rsc_logic, ["temperature"], {}, full=full)

    # Mock to check call occurs as expected with right return
    if full:
        # 1 - to calculate, 2 - on try/except for config_dict, 3 - on flagging
        assert repeating_day.call_count == 3
    else:
        # 1 - on try/except for config_dict, 2 - on flagging
        assert repeating_day.call_count == 2    