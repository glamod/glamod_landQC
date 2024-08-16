"""
Contains tests for humidity.py
"""
import numpy as np
import datetime as dt
from unittest.mock import patch, Mock
import pytest

import humidity

import common
import qc_utils as utils

def _setup_station() -> utils.Station:

    # set up the data
    temps = np.ma.arange(10)
    temps.mask = np.zeros(len(temps))
    dewps = np.ma.arange(10)-1
    dewps.mask = np.zeros(len(dewps))

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    dew_point_temperature = common.example_test_variable("dew_point_temperature", dewps)
    
    # make Station
    station = common.example_test_station(temperature)
    station.dew_point_temperature = dew_point_temperature

    # and build the times, simple hourly stuff
    datetimes = np.array([dt.datetime(2024, 1, 1, 12, 0) +
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

    assert config_dict["HUMIDITY"]["DPD"] == -utils.MDI


@pytest.mark.parametrize("config_dict", [{}, {"HUMIDITY" : {}}])
def test_get_repeating_dpd_threshold(config_dict):

    # set up the data
    temps = np.arange(75)
    dewps = np.arange(75) - 2.
    # use same array as in qc_utils test
    locs = np.array([0,
                     10, 11, 12, 13, 14, 15,
                     20, 21, 22, 23,
                     30, 31, 32, 33,
                     40, 41, 42,
                     50, 51, 52,
                     60, 61, 62,
                     70])
    # create the DPD=0
    dewps[locs] = temps[locs]

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    dew_point_temperature = common.example_test_variable("dew_point_temperature", dewps)

    humidity.get_repeating_dpd_threshold(temperature, dew_point_temperature, config_dict)

    assert config_dict["HUMIDITY"]["DPD"] == 9.0



# NOT TESTING PLOTTING

def test_super_saturation_check() -> None:

    station = _setup_station()
    # manually trigger the super saturation
    station.dew_point_temperature.data[:3] = station.temperature.data[:3]+1

    expected = np.array(["h", "h", "h", "", "", "", "", "", "", ""])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)
                                          
def test_super_saturation_check_w_mask() -> None:

    station = _setup_station()

    # mask the first 3 entries
    station.temperature.data = np.ma.masked_where(station.temperature.data <=3,
                                                  station.temperature.data)
    station.dew_point_temperature.data = np.ma.masked_where(station.temperature.data <= 3,
                                                            station.dew_point_temperature.data)
    station.dew_point_temperature.data[-3:] = station.temperature.data[-3:]+1
    expected = np.array(["", "", "", "", "", "", "", "h", "h", "h"])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)

                                          
def test_super_saturation_check_proportion() -> None:

    station = _setup_station()

    # of the 10 element array set over 40% as Super Saturated
    station.dew_point_temperature.data[:5] = station.temperature.data[:5]+1

    expected = np.array(["h", "h", "h", "h", "h", "h", "h", "h", "h", "h"])

    humidity.super_saturation_check(station, station.temperature, station.dew_point_temperature)

    np.testing.assert_array_equal(station.dew_point_temperature.flags, expected)


#def test_dew_point_depression_check() -> None:

    # streaks of length 5
#    config_dict = {"HUMIDITY" : {"DPD" : 5}}


@patch("humidity.super_saturation_check")
@patch("humidity.dew_point_depression_streak")
def test_read_hcc(supersat_check_mock: Mock,
                  dpd_check_mock: Mock) -> None:

    station = _setup_station()

    # Do the call
    humidity.hcc(station, {})

    # Mock to check call occurs as expected with right return
    supersat_check_mock.assert_called_once()
    dpd_check_mock.assert_called_once()

