"""
Contains tests for streaks.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock
from typing import Optional

import streaks
import qc_utils

import common

EXCESS_STREAK_STARTS_LENGTHS = {10: 3,
                                20: 3,
                                30: 3,
                                40: 3,
                                50: 3,
                                60: 3,
                                70: 3,
                                80: 3,
                                90: 3,
                                100: 3,
                                110: 3,
                                120: 3,
                                130: 3,
                                140: 3,
                                150: 3,
                                160: 3,
                                170: 3,
                                180: 3,
                                190: 3,
                                }



def _make_station(data: np.array, name: str, times: Optional[np.array] = None) -> qc_utils.Station:
    """
    Create an example station 

    :param array data: data array (presumed temperatures)
    :param str name: name for variable

    :returns: Station
    
    """
    obs_var = common.example_test_variable(name, data)

    station = common.example_test_station(obs_var, times)

    return station


# not testing plotting


def test_mask_calms() -> None:

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

    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])

    # make some streaks (set start index and length)
    data = common.generate_streaky_data(data, common.REPEATED_STREAK_STARTS_LENGTHS)

    station = _make_station(data, "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}

    streaks.get_repeating_streak_threshold(this_var, config_dict)

    assert config_dict["STREAK-temperature"]["Straight"] == 8


def test_get_excess_streak_threshold() -> None:

    all_data = np.array([])
    years = np.array([])
    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])

    # add a some years with no streaks
    for y in range(10):
        all_data = np.append(all_data, data)
        years = np.append(years, (2000*np.ones(data.shape[0]))+y)

    data = common.generate_streaky_data(data, EXCESS_STREAK_STARTS_LENGTHS)

    # Add some years with streaks
    for y in range(10, 15):
        all_data = np.append(all_data, data)
        years = np.append(years, (2000*np.ones(data.shape[0]))+y)

    station = _make_station(all_data, "temperature")
    this_var = getattr(station, "temperature")

    config_dict = {}

    streaks.get_excess_streak_threshold(this_var, years, config_dict)

    assert config_dict["STREAK-temperature"]["Excess"] == 0.0335


def test_repeating_value_nonwind() -> None:

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
    CD_straight = {"Straight" : 10}  # streaks of 10 or more identical values
    config_dict["STREAK-temperature"] = CD_straight

    streaks.repeating_value(this_var, station.times, config_dict)

    np.testing.assert_array_equal(this_var.flags, expected_flags)


def test_repeating_value_wind() -> None:

    # set up data
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data[180:] = 0
    data.mask = np.zeros(data.shape[0])
    station = _make_station(data, "wind_speed")
    this_var = getattr(station, "wind_speed")

    # make the streak (should ignore the length-20 one of zeros [calms] at the end)
    this_var.data[50:70] = -5
    expected_flags = np.array(["" for _ in data])
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
    data = np.ma.arange(0, 200, 0.1).astype(float)
    data.mask = np.zeros(data.shape[0])
    data.mask[10] = True
    flags = np.array(["" for _ in data])

    # add a year with no streaks
    all_data = np.ma.append(all_data, data)
    expected_flags = np.append(expected_flags, flags)
    years = np.append(years, 2000*np.ones(data.shape[0]))

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


# def test_day_repeat():
# def test_hourly_repeat():


@patch("streaks.get_repeating_streak_threshold")
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
    