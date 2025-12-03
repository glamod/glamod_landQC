"""
Contains tests for frequent.py
"""
import numpy as np
import datetime as dt
import pandas as pd
import pytest
from unittest.mock import (call, patch, mock_open, Mock)

import frequent
import utils
import setup

import common

def _setup_station(indata: np.ma.array) -> utils.Station:
    """Create a station object to hold the information enabling
    the QC test to be tested

    Parameters
    ----------
    indata : np.ma.array
        dummy temperature data

    Returns
    -------
    utils.Station
        Station with appropriate attributes
    """
    # set up the data
    indata.mask = np.zeros(len(indata))

    # make MetVars
    temperature = common.example_test_variable("temperature", indata)

    # make Station
    station = common.example_test_station(temperature)

    return station

# not testing plotting routines

def test_get_histogram() -> None:
    """Test histogram and bins as expected"""

    indata = np.repeat(np.ma.arange(10), 10)

    width, hist, bins = frequent.get_histogram(indata, "temperature")

    expected = np.append(np.append(np.zeros(5),
                                   np.full((10), 10)), np.zeros(3)).astype(int)

    np.testing.assert_array_equal(expected, hist)

    assert width == 1
    assert bins[0] == -5
    assert bins[-1] == 13  # exclusive upper done twice (indata, bins)


# using 200 as values, need >1200 to trigger the test
@pytest.mark.parametrize("bad_value, expected", ([400, []],
                                                 [1200, []],
                                                 [1201, [3]]))
def test_scan_histogram(bad_value: int,
                        expected: list) -> None:
    """Test that peaks are pulled out as expected"""

    bins = np.arange(8)  # test just central vs the rest
    hist = np.full(7, 200)  # all 200s (above min value)
    hist[3] = bad_value

    result = frequent.scan_histogram(hist, bins)

    assert len(result) == len(expected)
    assert sorted(result) == sorted(expected)


# using 200 as values, need >1200 to trigger the test
@pytest.mark.parametrize("bad_value, expected", ([400, []],
                                                 [1200, []],
                                                 [1201, [3]]))
def test_scan_histogram(bad_value: int,
                        expected: list) -> None:
    """Test that peaks are pulled out as expected"""

    bins = np.arange(8)  # test just central vs the rest
    # Average value still 200, but now in a sloped set of data
    #   which is more realistic
    hist = np.arange(50, 400, 50)
    hist[3] = bad_value

    result = frequent.scan_histogram(hist, bins)

    assert len(result) == len(expected)
    assert sorted(result) == sorted(expected)


def test_identify_values_no_data() -> None:
    """Test identification script if there's no data"""

    indata = np.ma.array([1])
    station = _setup_station(indata)
    obs_var = station.temperature

    config_dict = {}

    frequent.identify_values(obs_var, station, config_dict)

    for month in range(1, 13):

        assert config_dict["FREQUENT-temperature"][f"{month}-width"] == -1
        assert config_dict["FREQUENT-temperature"][f"{month}"] == []


def test_identify_values() -> None:
    """Test identification script and storage of results"""

    # 30 instances of 0..23
    indata = np.repeat(np.ma.arange(24), 30)
    # every 5th entry set to 10
    indata[::5] = 10
    station = _setup_station(indata)
    obs_var = station.temperature

    config_dict = {}

    frequent.identify_values(obs_var, station, config_dict)

    assert config_dict["FREQUENT-temperature"][f"1-width"] == "1.0"
    assert config_dict["FREQUENT-temperature"][f"1"] == [10]


def test_identify_values_gaussian() -> None:
    """Check identification works using gaussian data"""
    nyears = 10
    month_hours = 24*31
    # 10 years of Januaries at 0.1C resolution
    indata = np.ma.array(np.round(np.random.normal(15, 10,
                                                   month_hours * nyears),
                                  decimals=1))
    indata.mask = np.zeros(len(indata))
    # every 5th entry set to 0 (mean is 15)
    indata[::5] = 0

    # make MetVars
    temperature = common.example_test_variable("temperature", indata)

    for y in range(nyears):
        start_dt = dt.datetime(2000+y, 1, 1, 0, 0)
        month_times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                                for i in range(month_hours)])[0])
        if y == 0:
            times = month_times.copy()
        else:
            times = pd.concat([times, month_times])
    # check at this generation stage that have all Januaries
    assert np.unique(times.dt.month) == 1

    # make Station, by hand so can set times
    station = common.example_test_station(temperature, times)

    config_dict = {}

    frequent.identify_values(temperature, station, config_dict)

    # data resolution at 0.1 so width == 0.5
    assert config_dict["FREQUENT-temperature"][f"1-width"] == "0.5"
    assert config_dict["FREQUENT-temperature"][f"1"] == [0]


# def test_frequent_values() -> None:

# def test_fvc() -> None: