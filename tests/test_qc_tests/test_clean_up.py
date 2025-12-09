"""
Contains tests for clean_up.py
"""
import numpy as np
import pytest
from unittest.mock import patch, Mock

import clean_up

import common


def test_clean_up_all_flagged() -> None:

    # Have one month of data
    temperatures = common.example_test_variable("temperature", np.zeros(30*24))
    temperatures.flags = np.array(["l" for _ in range(temperatures.data.shape[0])])
    station = common.example_test_station(temperatures)

    new_flags = clean_up.clean_up(temperatures, station)

    # these are just the flags set by clean_up.py
    #  As all flagged, then although above the fraction, none reflagged
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])

    np.testing.assert_array_equal(new_flags, expected_flags)


def test_clean_up_low_count() -> None:

    # Have one month of data, but want low numbers of unflagged data, but small
    #   fraction flagged
    temperatures = common.example_test_variable("temperature", np.zeros(5))
    temperatures.flags = np.array(["l" for _ in range(temperatures.data.shape[0])])
    temperatures.flags[:3] = ""  # (3/5 unflagged, 2/5 flagged)
    station = common.example_test_station(temperatures)

    new_flags = clean_up.clean_up(temperatures, station, low_counts=5)

    # these are just the flags set by clean_up.py
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])
    expected_flags[:3] ="e"

    np.testing.assert_array_equal(new_flags, expected_flags)


def test_clean_up_none_flagged() -> None:

    # Have one month of data
    temperatures = common.example_test_variable("temperature", np.zeros(30*24))
    station = common.example_test_station(temperatures)

    new_flags = clean_up.clean_up(temperatures, station)

    # these are just the flags set by clean_up.py
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])

    np.testing.assert_array_equal(new_flags, expected_flags)


@pytest.mark.parametrize("fraction, expected", [(0.59, 0),
                                                 (0.60, 0),
                                                 (0.61, 1)])
def test_clean_up(fraction: float, expected: int) -> None:
    # dictionary to set the expected flags, given the passed fraction
    flag_value = {0: "", 1: "e"}

    # Have one month of data
    temperatures = common.example_test_variable("temperature", np.zeros(30*24))
    temperatures.flags = np.array(["" for _ in range(temperatures.data.shape[0])])
    # set a fraction of the flags
    temperatures.flags[:int(temperatures.data.shape[0]*fraction)] = "l"

    station = common.example_test_station(temperatures)

    new_flags = clean_up.clean_up(temperatures, station)

    # these are just the flags set by clean_up.py
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])
    expected_flags[int(temperatures.data.shape[0]*fraction):] = flag_value[expected]

    np.testing.assert_array_equal(new_flags, expected_flags)


@patch("clean_up.clean_up")
def test_mcu(clean_up_mock: Mock) -> None:

    temperatures = common.example_test_variable("temperature", np.arange(5))
    station = common.example_test_station(temperatures)

    expected_flags = np.array(["e" for _ in range(temperatures.data.shape[0])])
    clean_up_mock.return_value = expected_flags

    clean_up.mcu(station, ["temperature"])

    clean_up_mock.assert_called_once()
    np.testing.assert_array_equal(temperatures.flags,
                                  expected_flags)

    return
