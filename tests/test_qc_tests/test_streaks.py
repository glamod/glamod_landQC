"""
Contains tests for streaks.py
"""
import numpy as np
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import streaks
import qc_utils

import common

def _make_station(data: np.array, name: str) -> qc_utils.Station:
    """
    Create an example station 

    :param array data: data array (presumed temperatures)
    :param str name: name for variable

    :returns: Station
    
    """
    obs_var = common.example_test_variable(name, data)

    times = np.array([dt.datetime(2024, 1, 1, 12, 0) +
                     (i * dt.timedelta(seconds=60*60))
                     for i in range(len(data))])
    
    station = common.example_test_station(obs_var, times)

    return station


# not testing plotting


def test_mask_calms():

    # set up data
    data = np.ma.arange(2).astype(float)
    data.mask = np.zeros(data.shape[0])
    station = _make_station(data, "wind_speed")
    this_var = getattr(station, "wind_speed")

    # call routine
    streaks.mask_calms(this_var)

    assert this_var.data.mask[0] == True
    assert this_var.data.mask[1] == False


def test_get_repeating_string_threshold():

    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])

    # make some streaks (set start index and length)
    streak_starts_lengths = {10: 3,
                             20: 3,
                             30: 3,
                             40: 3,
                             50: 3,
                             60: 3,
                             70: 4,
                             80: 4,
                             90: 4,
                             100: 4,
                             110: 4,
                             120: 5,
                             130: 5,
                             140: 5,
                             150: 6,
                             160: 6,
                             170: 7,
                             }
    
    for start, length in streak_starts_lengths.items():
        data[start: start+length] = data[start]

    station = _make_station(data, "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}

    streaks.get_repeating_string_threshold(this_var, config_dict)

    assert config_dict["STREAK-temperature"]["Straight"] == 8


def test_repeating_value_nonwind():

    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])
    station = _make_station(data, "temperature")
    this_var = getattr(station, "temperature")

    # make the streak
    this_var.data[50:70] = -5
    expected_flags = np.array(["" for _ in data])
    expected_flags[50:70] = "K"

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : 10}  # strings of 10 or more identical values
    config_dict["STREAK-temperature"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_repeating_value_wind():

    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data[180:] = 0
    data.mask = np.zeros(data.shape[0])
    station = _make_station(data, "wind_speed")
    this_var = getattr(station, "wind_speed")

    # make the streak (should ignore the length-20 one at the end)
    this_var.data[50:70] = -5
    expected_flags = np.array(["" for _ in data])
    expected_flags[50:70] = "K"

    # set up dictionary
    config_dict = {}
    CD_straight = {"Straight" : 10}  # strings of 10 or more identical values
    config_dict["STREAK-wind_speed"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)




# def test_excess_repeating_value():
# def test_day_repeat():
# def test_hourly_repeat():


@patch("streaks.get_repeating_string_threshold")
@patch("streaks.repeating_value")
def test_pcc(repeating_value_mock: Mock,
             get_threshold_mock: Mock) -> None:

    # Set up data, variable & station
    temps = np.ma.arange(0, 100, 0.1)
    station = _make_station(temps, "temperature")


    # Do the call
    streaks.rsc(station, ["temperature"], {}, full=True)

    # Mock to check call occurs as expected with right return
    repeating_value_mock.assert_called_once()
    get_threshold_mock.assert_called_once()
    