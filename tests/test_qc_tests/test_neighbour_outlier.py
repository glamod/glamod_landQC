"""
Contains tests for neighour_outlier.py
"""
import numpy as np
import pandas as pd
import datetime as dt
import os
from unittest.mock import patch, Mock

import neighbour_outlier
import qc_utils
import io_utils

import common

EXAMPLE_FILES_PATH = os.path.join(os.path.dirname(__file__), "../", "example_data")

# not testing plotting


def _make_target_and_buddy(start_dt: dt.datetime | None = None) -> tuple[qc_utils.MeteorologicalVariable,
                                      qc_utils.MeteorologicalVariable]:

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
                ["NotBuddy", 45, 100, 10],
                ["Buddy", 45, 100, 10],
                ]
    station_list = pd.DataFrame(instring, columns=["id", "latitude",
                                                   "longitude", "elevation"])

    buddy_list = np.array([["Target", 0], ["Buddy", "20"],["-", "9999"]])

    return station_list, buddy_list


@patch("neighbour_outlier.utils.get_station_list")
@patch("neighbour_outlier.io_utils.read_station")
def test_read_in_buddies(read_station_mock: Mock,
                         get_station_list_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    get_station_list_mock.return_value = station_list
    read_station_mock.return_value = (buddy_station , None)

    buddy_dict = neighbour_outlier.read_in_buddies(target_station, buddy_list)

    # is buddy in the dictionary
    assert "Buddy" in buddy_dict.keys()
    # assert it is a Station with suitable values
    assert isinstance(buddy_dict["Buddy"], qc_utils.Station)
    assert buddy_dict["Buddy"].lon == 100
    np.testing.assert_array_equal(buddy_dict["Buddy"].temperature.data, buddy_station.temperature.data)


@patch("neighbour_outlier.utils.get_station_list")
@patch("neighbour_outlier.io_utils")
def test_read_in_buddies_oserror(io_utils_mock: Mock,
                                 get_station_list_mock: Mock) -> None:

    target_station, _ = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    get_station_list_mock.return_value = station_list

    io_utils_mock.read_station.side_effect = OSError
    _ = neighbour_outlier.read_in_buddies(target_station, buddy_list)

    # Check error handling
    io_utils_mock.write_error.assert_called_once_with(target_station,
                                                "File Missing (Buddy check): Buddy")


@patch("neighbour_outlier.utils.get_station_list")
@patch("neighbour_outlier.io_utils")
def test_read_in_buddies_valueerror(io_utils_mock: Mock,
                                    get_station_list_mock: Mock) -> None:

    target_station, _ = _make_target_and_buddy()
    station_list, buddy_list = _make_station_and_buddy_list()

    get_station_list_mock.return_value = station_list
    io_utils_mock.read_station.side_effect = ValueError("error text")
    _ = neighbour_outlier.read_in_buddies(target_station, buddy_list)

    # Check error handling
    io_utils_mock.write_error.assert_called_once_with(target_station,
                                                "Error in input file (Buddy check): Buddy",
                                                error="error text")


def test_read_in_buddy_data() -> None:

    target_station, buddy_station = _make_target_and_buddy()

    # add some extra ones to ensure resulting array is correct length
    buddy_list = np.array([["Target", 0],
                           ["Buddy1", "20"],
                           ["Buddy2", "20"],
                           ["Buddy3", "20"],
                           ["-", "9999"]])

    all_buddies = {"Buddy1" : buddy_station,
                   "Buddy2" : buddy_station,
                   "Buddy3" : buddy_station}

    # do the call
    buddy_data = neighbour_outlier.read_in_buddy_data(target_station, buddy_list,
                                                      all_buddies, "temperature")

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0
    assert len(buddy_data) == 5
    # last entry wasn't filled
    assert len(buddy_data[-1].compressed()) == 0


@patch("neighbour_outlier.io_utils.read_station")
def test_read_in_buddy_data_offset(read_station_mock: Mock) -> None:

    start_dt = dt.datetime(2000, 1, 1, 2, 0) # 2 hour offset
    target_station, buddy_station = _make_target_and_buddy(start_dt)
    _, buddy_list = _make_station_and_buddy_list()
    all_buddies = {"Buddy" : buddy_station}

    # Mock the return value
    read_station_mock.return_value = (buddy_station , None)

    # do the call
    buddy_data = neighbour_outlier.read_in_buddy_data(target_station, buddy_list,
                                                      all_buddies, "temperature")

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.array([None, None, 0, 1, 2, 3, 4, 5, 6, 7],
                                                              mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))



@patch("neighbour_outlier.io_utils.read_station")
def test_read_in_buddy_data_short(read_station_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    _, buddy_list = _make_station_and_buddy_list()
    all_buddies = {"Buddy" : buddy_station}

    # Mock return values
    read_station_mock.return_value = (buddy_station , None)

    # do the call
    buddy_data = neighbour_outlier.read_in_buddy_data(target_station, buddy_list,
                                                      all_buddies, "temperature")

    # first entry is blank (for self)
    assert len(buddy_data[0].compressed()) == 0

    # second should contain buddy data
    np.testing.assert_array_equal(buddy_data[1], np.ma.arange(10))


@patch("neighbour_outlier.io_utils.write_error")
def test_read_in_buddy_data_attributeerror(write_error_mock: Mock) -> None:

    target_station, buddy_station = _make_target_and_buddy()
    _, buddy_list = _make_station_and_buddy_list()
    all_buddies = {"Buddy" : buddy_station}


    _ = neighbour_outlier.read_in_buddy_data(target_station, buddy_list,
                                             all_buddies, "dummy_var")

        # Check error handling
    write_error_mock.assert_called_once_with(target_station,
                                            "Variable Missing (Buddy check): dummy_var - Buddy",
                                            error="'Station' object has no attribute 'dummy_var'")


def test_calculate_data_spread() -> None:

     # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(180))
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    # 4 hourly data so exceed min data count, but just for a single month and year
    times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=4*i)\
                           for i in range(len(temperatures.data))])[0])
    target_station = common.example_test_station(temperatures, times)

    # 4 buddies, range of differences but constant offset
    differences = np.ma.ones([4, 180])
    differences[:, 100:] *= 4

    spread = neighbour_outlier.calculate_data_spread(target_station, differences)

    # Just assessing that this month's value is correct
    np.testing.assert_array_equal(spread[0][0], qc_utils.spread(differences[0]))


def test_calculate_data_spread_short() -> None:

     # set up variables and stations
    temperatures = common.example_test_variable("temperature", np.arange(100))
    start_dt = dt.datetime(2000, 1, 1, 0, 0)

    # 4 hourly data so but insufficient length
    times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=4*i)\
                           for i in range(len(temperatures.data))])[0])
    target_station = common.example_test_station(temperatures, times)

    # 4 buddies, range of differences but constant offset
    differences = np.ma.ones([4, 100])
    differences[:, 50:] *= 4

    spread = neighbour_outlier.calculate_data_spread(target_station, differences)

    # Just assessing that the month's value is correct
    #   Short record, so MIN_SPREAD
    np.testing.assert_array_equal(spread[0][0], neighbour_outlier.MIN_SPREAD)


def test_adjust_pressure_for_tropical_storms_nearby() -> None:
    buddy_list = np.array([["Target", 0],
                           ["Buddy1", "80"],
                           ["Buddy2", "80"]])

    differences = np.ma.ones([3, 120]) # ~30d of 4hrly obs
    differences[0] = 0 # for target
    differences[1, :10] = 6
    differences[2, -10:] = 5
    spreads = np.ma.ones([3, 120]) # min spread is 1 in this test case
    spreads[0] = 0 #  for target

    dubious = np.ma.zeros(differences.shape)
    dubious = neighbour_outlier.adjust_pressure_for_tropical_storms(dubious,
                                                                    buddy_list,
                                                                    differences,
                                                                    spreads)

    expected = np.ma.zeros(differences.shape)
    expected[1, :10] = 1
    # second set are equal to, not greater than

    np.testing.assert_array_equal(expected, dubious)


def test_adjust_pressure_for_tropical_storms_none() -> None:
    buddy_list = np.array([["Target", 0],
                           ["Buddy1", "120"],
                           ["Buddy2", "120"],
                           ["Buddy3", "120"],
                           ["Buddy4", "80"]])

    differences = np.ma.ones([5, 120]) # ~30d of 4hrly obs
    differences[0] = 0 # for target
    differences[1:, :10] = -6  # set negative flags for all buddies
    differences[1:, -10:] = 6 # set some positive flags
    spreads = np.ma.ones([5, 120]) # min spread is 1 in this test case
    spreads[0] = 0 #  for target

    dubious = np.ma.zeros(differences.shape)
    dubious = neighbour_outlier.adjust_pressure_for_tropical_storms(dubious,
                                                                    buddy_list,
                                                                    differences,
                                                                    spreads)

    # For Buddy2, len(neg) = len(pos), so no flags
    # For Buddy4, not distant enough, so not affected by this test
    expected = np.ma.zeros(differences.shape)
    expected[1:4, :10] = 1
    expected[1:4, -10:] = 1

    np.testing.assert_array_equal(expected, dubious)


def test_adjust_pressure_for_tropical_storms() -> None:
    buddy_list = np.array([["Target", 0],
                           ["Buddy1", "120"],
                           ["Buddy2", "120"],
                           ["Buddy3", "120"],
                           ["Buddy4", "80"]])

    differences = np.ma.ones([5, 120]) # ~30d of 4hrly obs
    differences[0] = 0 # for target
    differences[1:, :30] = -6  # set negative flags for all buddies
    differences[2, -10:] = 6 # set some positive flags
    spreads = np.ma.ones([5, 120]) # min spread is 1 in this test case
    spreads[0] = 0 #  for target

    dubious = np.ma.zeros(differences.shape)
    dubious = neighbour_outlier.adjust_pressure_for_tropical_storms(dubious,
                                                                    buddy_list,
                                                                    differences,
                                                                    spreads)

    # For Buddy2, 75% negative flags, so highlight positives
    expected = np.ma.zeros(differences.shape)
    expected[2, -10:] = 1

    np.testing.assert_array_equal(expected, dubious)


@patch("neighbour_outlier.read_in_buddy_data")
def test_neighbour_outlier(read_buddy_data_mock: Mock) -> None:

    initial_neighbours = np.array([["Target", 0],
                                    ["Buddy1", "120"],
                                    ["Buddy2", "120"],
                                    ["Buddy3", "120"],
                                    ["Buddy4", "80"]])

    all_buddies = {"Buddy1" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy2" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy3" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy4" : qc_utils.Station("Buddy1", 45, 100 , 10),}

    temperatures = common.example_test_variable("temperature", np.arange(140))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    expected_flags = np.array(["" for i in range(temperatures.data.shape[0])])

    # Make the buddy data, initially the same as target, and then deviate
    all_buddy_data = np.tile(np.ma.arange(140.), (5, 1))
    all_buddy_data.mask = np.zeros(all_buddy_data.shape)
    all_buddy_data.mask[0, :] = True # target
    all_buddy_data[1:4, :] += 1. # simple offset

    # Use of simple offset means spreads == MIN_SPREAD
    # now set differences > SPREAD_LIMIT*MIN_SPREAD
    all_buddy_data[1:4, :20] += 20.
    expected_flags[:20] = "N"

    read_buddy_data_mock.return_value = all_buddy_data

    neighbour_outlier.neighbour_outlier(target_station, initial_neighbours, all_buddies, "temperature")

    np.testing.assert_array_equal(target_station.temperature.flags, expected_flags)


@patch("neighbour_outlier.read_in_buddy_data")
def test_neighbour_outlier_clean(read_buddy_data_mock: Mock) -> None:

    initial_neighbours = np.array([["Target", 0],
                                    ["Buddy1", "120"],
                                    ["Buddy2", "120"],
                                    ["Buddy3", "120"],
                                    ["Buddy4", "120"]])

    all_buddies = {"Buddy1" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy2" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy3" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy4" : qc_utils.Station("Buddy1", 45, 100 , 10),}

    temperatures = common.example_test_variable("temperature", np.arange(140))
    target_station = common.example_test_station(temperatures)
    target_station.id = "Target"
    expected_flags = np.array(["" for i in range(temperatures.data.shape[0])])

    # Make the buddy data, initially the same as target, and then deviate
    all_buddy_data = np.tile(np.ma.arange(140.), (5, 1))
    all_buddy_data.mask = np.zeros(all_buddy_data.shape)
    all_buddy_data.mask[0, :] = True # target
    all_buddy_data[1:2, :] += 1. # simple offset

    # Use of simple offset means spreads == MIN_SPREAD
    # now set differences > SPREAD_LIMIT*MIN_SPREAD
    all_buddy_data[1:2, :20] += 20.
    # Only 1 neighbour bad, so too small a fraction to set any flags

    read_buddy_data_mock.return_value = all_buddy_data

    neighbour_outlier.neighbour_outlier(target_station, initial_neighbours, all_buddies, "temperature")

    np.testing.assert_array_equal(target_station.temperature.flags, expected_flags)


@patch("neighbour_outlier.read_in_buddies")
@patch("neighbour_outlier.neighbour_outlier")
def test_noc(neighbour_outlier_mock: Mock,
             read_in_buddies_mock: Mock) -> None:

    # Set up data, variable & station
    temperatures = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(temperatures)

    read_in_buddies_mock.return_value  = {"Buddy1" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy2" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy3" : qc_utils.Station("Buddy1", 45, 100 , 10),
                   "Buddy4" : qc_utils.Station("Buddy1", 45, 100 , 10),}

    # Do the call, with 4 neighbours
    neighbour_outlier.noc(station, np.array([["Buddy1", 20],
                                             ["Buddy2", 30],
                                             ["Buddy3", 40],
                                             ["Buddy4", 50],]), ["temperature"])

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


@patch("neighbour_outlier.utils.get_station_list")
@patch("neighbour_outlier.setup")
def test_noc_example_data(setup_mock: Mock,
                          station_list_mock: Mock) -> None:

    # the neighbours and target all have data for 1979.

    setup_mock.SUBDAILY_PROC_DIR = EXAMPLE_FILES_PATH
    setup_mock.OUT_COMPRESSION = ""

    station_list = [["CHM00052353", 41.967, 100.883, 946.0],
                ["AJM00037843", 40.0170, 48.4670, -15.0],
                ["AJM00037849", 40.0170, 48.9170, -5.0],
                ["AJM00037898", 39.6500, 46.5330, 1099.0],
                ["AMM00037874", 39.8000, 44.7000, 818.0], # no overlapping data
                ]
    station_list_mock.return_value = pd.DataFrame(station_list, columns=["id", "latitude",
                                                   "longitude", "elevation"])

    target_station = qc_utils.Station("CHM00052353", 41.967, 100.883, 946.0)
    # TODO: update station filename/id/data after Release 8 [Sept 2025]
    target_station, _ = io_utils.read_station(os.path.join(EXAMPLE_FILES_PATH, "CHM00052353.qff2"),
                                              target_station, read_flags=True)

    initial_neighbours = np.array([["CHM00052353", 0],
                                  ["AJM00037843", 110],
                                  ["AJM00037849", 120],
                                  ["AJM00037898", 130],
                                  ["AMM00037874", 140]])

    var_list = ["sea_level_pressure"]

    neighbour_outlier.noc(target_station, initial_neighbours, var_list, diagnostics=True)

    # Manually set bad values (positive) 1979/8/14 and negative 1979/8/15
    locs, = np.nonzero(np.logical_and(np.logical_and(target_station.years == 1979,
                                                     target_station.months == 8),
                                      target_station.days == 14))
    example_flags = np.array(["" for _ in target_station.times])
    example_flags[locs] = "N"

    print(target_station.sea_level_pressure.flags)
    print(locs)
    np.testing.assert_array_equal(target_station.sea_level_pressure.flags, example_flags)
