"""
Contains tests for neighour_outlier.py
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import neighbour_outlier
import qc_utils

import common

# not testing plotting

@patch("neighbour_outlier.io.read_station")
def test_read_in_buddies_short(read_station_mock: Mock) -> None:

    # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(10))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    buddy_station = common.example_test_station(temperatures)
    buddy_station.id = "Buddy"

    read_station_mock.return_value = (buddy_station , None)

    instring = [["Target", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude", "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", 20]])

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.arange(10))


@patch("neighbour_outlier.io.read_station")
def test_read_in_buddies(read_station_mock: Mock) -> None:

    # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(10))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    buddy_station = common.example_test_station(temperatures)
    buddy_station.id = "Buddy"

    read_station_mock.return_value = (buddy_station , None)

    instring = [["Target", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude", "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", 20]])

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.arange(10))

@patch("neighbour_outlier.io")
def test_read_in_buddies_oserror(io_mock: Mock) -> None:

    # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(10))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    buddy_station = common.example_test_station(temperatures)
    buddy_station.id = "Buddy"

    io_mock.read_station.side_effect = OSError

    instring = [["Target", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude", "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", 20]])

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # Check error handling
    io_mock.write_error.assert_called_once_with(target_station,
                                                "File Missing (Buddy, temperature) - Buddy")


@patch("neighbour_outlier.io")
def test_read_in_buddies_valueerror(io_mock: Mock) -> None:

    # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(10))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    buddy_station = common.example_test_station(temperatures)
    buddy_station.id = "Buddy"

    io_mock.read_station.side_effect = ValueError("error text")

    instring = [["Target", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude", "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", 20]])

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # Check error handling
    io_mock.write_error.assert_called_once_with(target_station,
                                                "Error in input file (Buddy, temperature) - Buddy",
                                                error="error text")


#def test_calculate_data_spread() -> None:


#def test_adjust_pressure_for_tropical_storms() -> None:


#def test_neighbour_outlier() -> None:



@patch("neighbour_outlier.neighbour_outlier")
def test_noc(neighbour_outlier_mock: Mock) -> None:

    # Set up data, variable & station
    temperatures = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(temperatures)

    # Do the call, with 4 neighbours
    neighbour_outlier.noc(station, np.array([["ID1", 20],
                                             ["ID2", 30],
                                             ["ID3", 40],
                                             ["ID4", 50],]), ["temperature"])

    # Mock to check call occurs as expected with right return
    neighbour_outlier_mock.assert_called_once()


@patch("neighbour_outlier.neighbour_outlier")
def test_noc_few_buddies(neighbour_outlier_mock: Mock) -> None:

    # Set up data, variable & station
    temperatures = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(temperatures)

    # Do the call with one neighbour
    neighbour_outlier.noc(station, np.array([["ID", 20]]), ["temperature"])

    # Mock to check call wasn't made
    neighbour_outlier_mock.assert_not_called()