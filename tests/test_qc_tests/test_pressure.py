"""
Contains tests for pressure.py
"""
import numpy as np
import datetime as dt
from unittest.mock import patch, Mock

import pressure

import common


# not testing plots

def test_identify_values():

    # Set up data, variables &c
    test_data = np.arange(150)
    expected_average = 10

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)

    test_data -= expected_average
    # no variation in spread

    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    
    config_dict = {}
    pressure.identify_values(sealp, stnlp, config_dict)

    assert config_dict["PRESSURE"]["average"] == expected_average
    assert config_dict["PRESSURE"]["spread"] == pressure.MIN_SPREAD

def test_pressure_offset_low():

    # Set up data, variables &c
    test_data = np.arange(150)
    expected_flags = np.array(["" for _ in test_data])
    expected_flags[:10] = "p"

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)

    test_data -= 1
    test_data[:10] -= 30
    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    times = np.array([start_dt + dt.timedelta(hours=i)\
                      for i in range(len(sealp.data))])
    
    config_dict = {"PRESSURE": {"average": 1, "spread": 1}}

    pressure.pressure_offset(sealp, stnlp, times, config_dict)

    np.testing.assert_array_equal(stnlp.flags, expected_flags)


def test_pressure_offset_high():

    # Set up data, variables &c
    test_data = np.arange(150)
    expected_flags = np.array(["" for _ in test_data])
    expected_flags[:10] = "p"

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)

    test_data -= 1
    test_data[:10] += 30
    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    times = np.array([start_dt + dt.timedelta(hours=i)\
                      for i in range(len(sealp.data))])
    
    config_dict = {"PRESSURE": {"average": 1, "spread": 1}}

    pressure.pressure_offset(sealp, stnlp, times, config_dict)

    np.testing.assert_array_equal(stnlp.flags, expected_flags)


def test_pressure_offset_bad_population():

    # Set up data, variables &c
    test_data = np.arange(150)
    expected_flags = np.array(["" for _ in test_data])

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)

    # set large fraction to high offset
    test_data[:50] += 30
    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    times = np.array([start_dt + dt.timedelta(hours=i)\
                      for i in range(len(sealp.data))])
    
    config_dict = {"PRESSURE": {"average": 1, "spread": 1}}

    pressure.pressure_offset(sealp, stnlp, times, config_dict)

    np.testing.assert_array_equal(stnlp.flags, expected_flags)


def test_calc_slp_0m():

    stnlp = np.ma.arange(790, 900, 10)
    stnlp.mask = np.zeros(10)

    elevation = 0

    temperature = np.ma.arange(0, 22, 2)
    temperature.mask = np.zeros(10)

    sealp = pressure.calc_slp(stnlp, elevation, temperature)

    print(sealp)
    np.testing.assert_array_equal(stnlp, sealp)


def test_calc_slp_1566m():

    stnlp = np.ma.array([790])
    stnlp.mask = np.zeros(1)

    elevation = 1566

    temperature = np.ma.array([10])
    temperature.mask = np.zeros(1)

    sealp = pressure.calc_slp(stnlp, elevation, temperature)

    expected_sealp = np.ma.array([951.176798])
    # checked result by hand 7-Aug-2023
    np.testing.assert_array_almost_equal(expected_sealp, sealp)


def adjust_existing_flag_locs():

    test_data = np.arange(10)
    test_data[:2] = -99
    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data, mdi=-99)
    
    # first two entries have flags
    sealp.flags = np.array(["" for _ in test_data])
    sealp.flags[-2:] = "p"

    # wanting to set flags on all entries
    set_flags = np.array(["p" for _ in test_data])
    
    # so expect to have none set on first two entries of adjusted array
    expected_flags = np.array(["p" for _ in test_data])
    expected_flags[-2:] = ""
    
    new_flags = pressure.adjust_existing_flag_locs(sealp, set_flags)

    np.testing.assert_array_equal(new_flags, expected_flags)


def test_pressure_theory_0m():

    # Set up data, variables & station
    test_data = np.arange(6)

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)
    sealp.flags = np.array(["" for i in test_data])

    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    stnlp.flags = np.array(["" for i in test_data])

    test_data[5] = -99
    temperature = common.example_test_variable("temperature",
                                               test_data, mdi=-99)
    temperature.flags = np.array(["" for i in test_data])

    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    times = np.array([start_dt + dt.timedelta(hours=i)\
                      for i in range(len(sealp.data))])

    pressure.pressure_theory(sealp, stnlp, temperature, times, 0)


    np.testing.assert_array_equal(stnlp.flags, np.array(["" for i in test_data]))
    np.testing.assert_array_equal(sealp.flags, np.array(["" for i in test_data]))

  
@patch("pressure.identify_values")
@patch("pressure.pressure_offset")
@patch("pressure.pressure_theory")
def test_pcc(pressure_theory_mock: Mock,
             pressure_offset_mock: Mock,
             identify_values_mock: Mock):

    # Set up data, variables & station
    test_data = np.arange(5)

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)
    sealp.flags = np.array(["" for i in test_data])

    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    stnlp.flags = np.array(["" for i in test_data])

    temperature = common.example_test_variable("temperature",
                                               test_data)
    temperature.flags = np.array(["" for i in test_data])

    station = common.example_test_station(sealp)
    setattr(station, "station_level_pressure", stnlp)
    setattr(station, "temperature", temperature)

    # Do the call
    pressure.pcc(station, dict(), full=True)

    # Mock to check call occurs as expected with right return
    pressure_offset_mock.assert_called_once()
    pressure_theory_mock.assert_called_once()
    identify_values_mock.assert_called_once()


@patch("pressure.identify_values")
@patch("pressure.pressure_offset")
@patch("pressure.pressure_theory")
def test_pcc_bad_elevation(pressure_theory_mock: Mock,
                           pressure_offset_mock: Mock,
                           identify_values_mock: Mock):

    # Set up data, variables & station
    test_data = np.arange(5)

    sealp = common.example_test_variable("sea_level_pressure",
                                         test_data)
    sealp.flags = np.array(["" for i in test_data])

    stnlp = common.example_test_variable("station_level_pressure",
                                         test_data)
    stnlp.flags = np.array(["" for i in test_data])

    temperature = common.example_test_variable("temperature",
                                               test_data)
    temperature.flags = np.array(["" for i in test_data])

    station = common.example_test_station(sealp, elevation=-999)
    setattr(station, "station_level_pressure", stnlp)
    setattr(station, "temperature", temperature)

    # Do the call
    pressure.pcc(station, dict(), full=True)

    # Mock to check call occurs as expected with right return
    identify_values_mock.assert_called_once()
    pressure_offset_mock.assert_called_once()
    assert not pressure_theory_mock.called
