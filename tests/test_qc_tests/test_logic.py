"""
Contains tests for logic.py
"""
import os
import numpy as np
import datetime as dt
import json
import pytest
from unittest.mock import (call, patch, mock_open, Mock)

import logic_checks
import qc_utils
import setup

import common


def _make_station() -> qc_utils.Station:
    """
    Create a station with two paired variables

    :returns: station
    
    """
    test_data = np.ones(10)
    obs_var = common.example_test_variable("temperature", test_data)
    station = common.example_test_station(obs_var)
    station.id = "DUMMY"
    
    return station


def test_logic_file() -> None:
    # check necessary keys in the JSON file and read structure
    with open(qc_utils.LOGICFILE, "r") as lf:
        limits = json.load(lf)["logic_limits"]

    for key in ["temperature",
                "dew_point_temperature",
                "sea_level_pressure",
                "station_level_pressure",
                "wind_speed",
                "wind_direction",
                ]:

        assert key in limits


@pytest.mark.parametrize("value", ([80, -80]))
def test_logic_check_large_fraction(value: int) -> None:

    index = 1 + int(1000*(logic_checks.BAD_THRESHOLD)) # how many bad values (0.6%)
    test_data = np.arange(0, 10, 0.01)
    test_data[:index] = value # lots too high/low
    expected = np.array(["" for _ in test_data])
    expected[:index] = "L"
    obs_var = common.example_test_variable("temperature", test_data)

    flags = logic_checks.logic_check(obs_var)

    np.testing.assert_array_equal(flags, expected)


@pytest.mark.parametrize("value", ([80, -80]))
def test_logic_check_small_fraction(value: int) -> None:

    index = int(1000*(logic_checks.BAD_THRESHOLD)) # 0.5% bad
    test_data = np.arange(0, 10, 0.01)
    test_data[:index] = value # lots too high/low
    expected = np.array(["" for _ in test_data])

    obs_var = common.example_test_variable("temperature", test_data)

    flags = logic_checks.logic_check(obs_var)

    np.testing.assert_array_equal(flags, expected)


def test_write_logic_error() -> None:
    # check writing being called as expected

    station = _make_station()

    # Mock the "open" aspect of filehandling
    open_mock = mock_open()
    with patch("logic_checks.open", open_mock, create=True):
        logic_checks.write_logic_error(station, "TEST MESSAGE")

        
    # Check output file name and path
    expected_dir = os.path.join(setup.SUBDAILY_ERROR_DIR, "DUMMY.err")
    open_mock.assert_called_with(expected_dir, "a")       
        
    # Assert the number of write calls is correct
    assert open_mock.return_value.write.call_count == 2
    
    # Assert the values of the write calls are correct
    calls = [call((dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M") + "\n")),
             call('TEST MESSAGE\n')
    ]
    open_mock.return_value.write.assert_has_calls(calls)


@pytest.mark.parametrize("latitude, expected", [(-91., -1),
                                                (-90., 0),
                                                (90., 0),
                                                (91., -1)])
@patch("logic_checks.write_logic_error")
def test_lc_latitude(write_error_mock: Mock, 
                     latitude: float,
                     expected: int) -> None:

    station = _make_station()
    station.lat = latitude

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected


@pytest.mark.parametrize("longitude, expected", [(-181., -1),
                                                 (-180., 0),
                                                 (180., 0),
                                                 (181., -1)])
@patch("logic_checks.write_logic_error")
def test_lc_longitude(write_error_mock: Mock, 
                      longitude: float,
                      expected: int) -> None:
    
    station = _make_station()
    station.lon = longitude

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected


@pytest.mark.parametrize("latitude, longitude, expected", [(-91., 181., -1),
                                                           (-91, -180., -1),
                                                           (-90, -181., -1),
                                                           (-90, -180., 0),
                                                           (0, 0., -1),
                                                           (90, 180., 0),
                                                           (90, 181., -1),
                                                           (91, 180., -1),
                                                           (91, 181., -1)])
@patch("logic_checks.write_logic_error")
def test_lc_combination(write_error_mock: Mock, 
                        latitude: float,
                        longitude: float,
                        expected: int) -> None:
    
    station = _make_station()
    station.lat = latitude
    station.lon = longitude

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected



@pytest.mark.parametrize("elevation, expected", [(-432.66, -1),
                                                (-432.65, 0),
                                                (0, 0),
                                                (-999, 0),
                                                (9999, 0),
                                                (8850, 0),                                                
                                                (8850.1, -1)])
@patch("logic_checks.write_logic_error")
def test_lc_elevation(write_error_mock: Mock, 
                      elevation: float,
                      expected: int) -> None:


    station = _make_station()
    station.elev = elevation

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected


@pytest.mark.parametrize("time, expected", [(dt.datetime(1699, 12, 31, 23, 0), -1),
                                            (dt.datetime(1700, 1, 1, 0, 0), 0),
                                            (dt.datetime(1700, 1, 1, 1, 0), 0),
                                            ])
@patch("logic_checks.write_logic_error")
def test_lc_start(write_error_mock: Mock, 
                  time: float,
                  expected: int) -> None:
    

    station = _make_station()
    station.times[0] = time
    print(station.times)

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected


@pytest.mark.parametrize("time, expected", [(dt.datetime.now() + dt.timedelta(hours=1), -1),
                                            (dt.datetime.now(), 0),
                                            ])
@patch("logic_checks.write_logic_error")
def test_lc_end(write_error_mock: Mock, 
                time: float,
                expected: int) -> None:
    

    station = _make_station()
    station.times[-1] = time

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == expected


@patch("logic_checks.write_logic_error")
def test_lc_time_diff(write_error_mock: Mock) -> None:
    

    station = _make_station()
    # repeat the first two entries
    station.times[8:] = station.times[:2]

    return_code = logic_checks.lc(station, ["temperature"])

    assert return_code == -1
