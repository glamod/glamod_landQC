"""
Contains tests for world_records.py
"""
import numpy as np
from unittest.mock import patch, Mock

import world_records

import common

@patch("world_records.record_check")
def test_read_wrc(record_check_mock: Mock):

    # Set up data, variable & station
    test_data = np.arange(5)

    this_var = common.example_test_variable("temperature",
                                            test_data)
    this_var.flags = np.array(["" for i in test_data])

    station = common.example_test_station(this_var)
    station.continent="row"

    # Set up flags to uses mocked return
    test_flags = np.array(["W" for i in test_data])
    record_check_mock.return_value = test_flags

    # Do the call
    world_records.wrc(station, ["temperature"])

    # Mock to check call occurs as expected with right return
    record_check_mock.assert_called_once()
    np.testing.assert_array_equal(station.temperature.flags, test_flags)


def test_record_check_nolocation():

    # Set up test data, with value to flag, and variable
    test_data = np.arange(5)
    test_data[0] = 60

    this_var = common.example_test_variable("temperature",
                                            test_data)

    # Do the call
    flags = world_records.record_check(this_var, "none")

    # Does the right value get flagged
    np.testing.assert_array_equal(flags, np.array(["W", "", "", "", ""]))


def test_record_check_africa():

    # Set up test data, with value to flag, and variable
    test_data = np.arange(5)
    test_data[0] = 115

    this_var = common.example_test_variable("wind_speed",
                                            test_data)

    # Do the call
    flags = world_records.record_check(this_var, "africa")

    # Does the right value get flagged
    np.testing.assert_array_equal(flags, np.array(["W", "", "", "", ""]))


def test_record_check_europe():

    # Set up test data, with value to flag, and variable
    test_data = np.arange(5)
    test_data[0] = 56.0

    this_var = common.example_test_variable("dew_point_temperature",
                                            test_data)

    # Do the call
    flags = world_records.record_check(this_var, "europe")

    # Does the right value get flagged
    np.testing.assert_array_equal(flags, np.array(["W", "", "", "", ""]))


def test_record_check_samerica():

    # Set up test data, with value to flag, and variable
    test_data = np.arange(5)
    test_data[0] = 1000

    this_var = common.example_test_variable("sea_level_pressure",
                                            test_data)

    # Do the call
    flags = world_records.record_check(this_var, "europe")

    # Does the right value get flagged
    np.testing.assert_array_equal(flags, np.array(["", "W", "W", "W", "W"]))
