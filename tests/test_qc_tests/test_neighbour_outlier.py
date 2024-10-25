"""
Contains tests for neighour_outlier.py
"""
import numpy as np
import pandas as pd
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import neighbour_outlier
import qc_utils

import common

# not testing plotting


def _make_target_and_buddy(start_dt: dt.datetime | None = None) -> tuple[qc_utils.Meteorological_Variable,
                                      qc_utils.Meteorological_Variable]:

    # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(10))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"

    if start_dt == None:
        times = None
    else:
        times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                           for i in range(len(temperatures.data))])[0])        

    buddy_station = common.example_test_station(temperatures, times)
    buddy_station.id = "Buddy"

    return target_station, buddy_station


def _make_station_and_buddy_list() -> tuple[pd.DataFrame, np.ndarray]:

    instring = [["Target", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude",
                                                   "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", "20"]])

    return station_list, buddy_list


@patch("neighbour_outlier.io.read_station")
def test_read_in_buddies_short(read_station_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    read_station_mock.return_value = (buddy_station , None)

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature")

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.arange(10))
    

@patch("neighbour_outlier.io.read_station")
def test_read_in_buddies_offset(read_station_mock: Mock) -> None:

    start_dt = dt.datetime(2000, 1, 1, 2, 0) # 2 hour offset
    target_station, buddy_station = _make_target_and_buddy(start_dt)
    station_list, buddy_list = _make_station_and_buddy_list()
    
    read_station_mock.return_value = (buddy_station , None)

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.array([None, None, 0, 1, 2, 3, 4, 5, 6, 7], 
                                                              mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))


@patch("neighbour_outlier.io.read_station")
def test_read_in_buddies(read_station_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    # add some extra ones to ensure resulting array is correct length
    buddy_list = np.append(buddy_list, [buddy_list[-1]], axis=0)
    buddy_list = np.append(buddy_list, [buddy_list[-1]], axis=0)

    read_station_mock.return_value = (buddy_station , None)

    buddy_data = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0
    assert len(buddy_data) == 4


@patch("neighbour_outlier.io")
def test_read_in_buddies_oserror(io_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()
    
    io_mock.read_station.side_effect = OSError

    _ = neighbour_outlier.read_in_buddies(target_station, station_list,
                                                   buddy_list, "temperature", diagnostics=True)

    # Check error handling
    io_mock.write_error.assert_called_once_with(target_station,
                                                "File Missing (Buddy, temperature) - Buddy")


@patch("neighbour_outlier.io")
def test_read_in_buddies_valueerror(io_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    io_mock.read_station.side_effect = ValueError("error text")

    _ = neighbour_outlier.read_in_buddies(target_station, station_list,
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