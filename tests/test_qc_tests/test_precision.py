"""
Contains tests for precision.py
"""
import numpy as np
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import precision
import qc_utils

import common

# not testing plotting
all_flagged = np.array(["n" for _ in range(120)])
none_flagged = np.array(["" for _ in range(120)])

def _make_station(temps: np.array, dewps: np.array) -> qc_utils.Station:
    """
    Create a station with two paired variables

    :returns: tuple(station)
    
    """
    primary = common.example_test_variable("temperature", temps)
    secondary = common.example_test_variable("dew_point_temperature", dewps)

    times = np.array([dt.datetime(2024, 1, 1, 12, 0) +
                     (i * dt.timedelta(seconds=60*60))
                     for i in range(len(dewps))])
    
    station = common.example_test_station(primary)
    station.dew_point_temperature = secondary
    common.add_times_to_example_station(station, times)

    return station


@pytest.mark.parametrize("t_divisor, d_divisor, expected", [(10, 1, all_flagged),
                                                             (10, 10, none_flagged)])
def test_precision_cross_check(t_divisor: int, d_divisor: int, expected: np.array):

    length = 120
    temps = np.ma.arange(0, length/t_divisor, 1/t_divisor)
    temps.mask = np.zeros(temps.shape[0])
    dewps = np.ma.arange(0, length/d_divisor, 1/d_divisor)
    dewps.mask = np.zeros(dewps.shape[0])
 
    station = _make_station(temps, dewps)

    precision.precision_cross_check(station,
                                    station.temperature,
                                    station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


def test_precision_cross_check_short_record():

    length = 100
    expected = np.array(["" for _ in range(length)])
 
    temps = np.ma.arange(0, length/10, 1/10)
    temps.mask = np.zeros(temps.shape[0])
    dewps = np.ma.arange(0, length, 1)
    dewps.mask = np.zeros(dewps.shape[0])
    
    station = _make_station(temps, dewps)

    precision.precision_cross_check(station,
                                    station.temperature,
                                    station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


@patch("precision.precision_cross_check")
def test_pcc(cross_check_mock: Mock) -> None:

    # Set up data, variable & station
    length = 120
    temps = np.ma.arange(0, length)
    dewps = np.ma.arange(0, length)

    station = _make_station(temps, dewps)

    # Set up flags to uses mocked return
    cross_check_mock.return_value = True

    # Do the call
    precision.pcc(station, {})

    # Mock to check call occurs as expected with right return
    cross_check_mock.assert_called_once()