"""
Contains tests for pressure.py
"""
import numpy as np
from unittest.mock import patch, Mock

import pressure

import common

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

#def test_pressure_theory():


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
    np.testing.assert_array_equal(stnlp, sealp)


#def test_pressure_offset():


#def test_identify_values():
