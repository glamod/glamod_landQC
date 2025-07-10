"""
Contains tests for timestamp.py
"""
import numpy as np
from unittest.mock import patch, Mock

import timestamp

import common

# not testing plotting

def test_identify_multiple_values_simple():

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(obs_var)

    station.times[3] = station.times[2]
    expected = np.array(["" for _ in range(10)])

    # Two time stamps the same, and values different, so flags set
    expected[2:4] = "T"

    timestamp.identify_multiple_values(obs_var, station. times, {})

    np.testing.assert_array_equal(obs_var.flags, expected)


def test_identify_no_multiple_values():

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(obs_var)

    expected = np.array(["" for _ in range(10)])

    timestamp.identify_multiple_values(obs_var, station. times, {})

    np.testing.assert_array_equal(obs_var.flags, expected)


def test_identify_multiple_values_masked_simple():

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(obs_var)

    station.times[4] = station.times[3] = station.times[2]
    obs_var.data.mask[4] = True
    expected = np.array(["" for _ in range(10)])

    # Three time stamps the same, and values different, so flags set
    #   But final one has data masked, so no flag included for that one.
    expected[2:4] = "T"

    timestamp.identify_multiple_values(obs_var, station. times, {})

    np.testing.assert_array_equal(obs_var.flags, expected)


def test_identify_multiple_values_masked():

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(obs_var)

    station.times[4] = station.times[3] = station.times[2]
    obs_var.data.mask[3] = True
    expected = np.array(["" for _ in range(10)])

    # Three time stamps the same, and values different, so flags set
    #   But middle one has data masked, so no flag included for that one.
    expected[2] = "T"
    expected[4] = "T"

    timestamp.identify_multiple_values(obs_var, station. times, {})

    np.testing.assert_array_equal(obs_var.flags, expected)
    
    
def test_identify_multiple_values_same():

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.ones(10))
    station = common.example_test_station(obs_var)

    station.times[3] = station.times[2]

    # Even though two time stamps are the same, the values are identical
    #   (np.ones) but an information flag still set
    expected = np.array(["" for _ in range(10)])
    expected[2:4] = "2"

    timestamp.identify_multiple_values(obs_var, station.times, {})

    np.testing.assert_array_equal(obs_var.flags, expected)


@patch("timestamp.identify_multiple_values")
def test_tsc(multiple_values_mock: Mock) -> None:

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.ones(10))
    station = common.example_test_station(obs_var)

    # Set up flags to uses mocked return
    multiple_values_mock.return_value = True

    # Do the call
    timestamp.tsc(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    multiple_values_mock.assert_called_once()
