"""
Contains tests for winds.py
"""
import numpy as np
from unittest.mock import patch, Mock

import winds

import common


def test_logical_checks_zero_direction() -> None:
    
    speed_data = np.array([0, 0, 10, 10, 10])
    direction_data = np.ma.array([0, 0, 90, 180, 360])
    direction_data.mask = np.zeros(direction_data.shape[0])
    direction_data.mask[1] = True

    # if not fixing, then setting flag
    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_direction_flags, directions.flags)

    return


def test_logical_checks_zero_direction_fix() -> None:
    
    speed_data = np.array([0, 0, 10, 10, 10])
    direction_data = np.ma.array([0, 360, 90, 180, 360])
    direction_data.mask = np.zeros(direction_data.shape[0])
    direction_data.mask[1] = True

    # if fixing, no flag set, calm set to 0N, mask undone
    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_data = np.ma.array([0, 0, 90, 180, 360])
    expected_direction_mask = np.zeros(direction_data.shape[0])

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions, fix=True)

    np.testing.assert_array_equal(expected_direction_flags, directions.flags)
    np.testing.assert_array_equal(expected_direction_data, directions.data)
    np.testing.assert_array_equal(expected_direction_mask, directions.data.mask)

    return


def test_logical_checks_negative_speed() -> None:
    
    speed_data = np.array([0, -10, 10, 10, 10])
    direction_data = np.array([0, 90, 90, 180, 360])
    expected_speed_flags = np.array(["" for _ in speed_data])
    expected_speed_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_speed_flags, speeds.flags)

    return


def test_logical_checks_negative_direction() -> None:
    
    speed_data = np.array([0, 10, 10, 10, 10])
    direction_data = np.array([0, -90, 90, 180, 360])
    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_direction_flags, directions.flags)

    return


def test_logical_checks_wrapped_direction() -> None:
    
    speed_data = np.array([0, 10, 10, 10, 10])
    direction_data = np.array([0, 420, 90, 180, 360])
    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_direction_flags, directions.flags)

    return


def test_logical_checks_bad_direction() -> None:
    
    speed_data = np.array([0, 0, 10, 10, 10])
    direction_data = np.array([0, 90, 90, 180, 360])
    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_direction_flags, directions.flags)

    return


def test_logical_checks_bad_speed() -> None:
    
    speed_data = np.array([0, 10, 10, 10, 10])
    direction_data = np.array([0, 0, 90, 180, 360])
    expected_speed_flags = np.array(["" for _ in speed_data])
    expected_speed_flags[1] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_speed_flags, speeds.flags)

    return


def test_logical_checks_all() -> None:

    # check all work in combination with good data    
    speed_data = np.array([0, 10, 20, 30, 40,
                           10, 0, 10, 10, -10, 0])
    direction_data = np.ma.array([0, 90, 180, 270, 360,
                               0, 10, 361, -1, 90, 1])
    direction_data.mask = np.zeros(direction_data.shape[0])
    direction_data.mask[-1] = True
    
    expected_speed_flags = np.array(["" for _ in speed_data])
    expected_speed_flags[5] = "w"
    expected_speed_flags[9] = "w"

    expected_direction_flags = np.array(["" for _ in speed_data])
    expected_direction_flags[6] = "w"
    expected_direction_flags[7] = "w"
    expected_direction_flags[8] = "w"
    expected_direction_flags[10] = "w"

    speeds = common.example_test_variable("wind_speed", speed_data)
    directions = common.example_test_variable("wind_direction", direction_data)

    winds.logical_checks(speeds, directions)

    np.testing.assert_array_equal(expected_speed_flags, speeds.flags)
    np.testing.assert_array_equal(expected_direction_flags, directions.flags)

    return


@patch("winds.logical_checks")
def test_wcc(logical_checks_mock: Mock) -> None:

    speeds = common.example_test_variable("wind_speed", np.arange(5))
    directions = common.example_test_variable("wind_direction", np.arange(5))
    station = common.example_test_station(speeds)
    station.wind_direction  = directions

    winds.wcc(station, {})

    logical_checks_mock.assert_called_once()


    return