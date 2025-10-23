"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock

import odd_cluster

import common
import utils

# not testing plotting

# generate some (masked) to pass into the tests
INDATA = np.ma.ones(50) # not testing values, only temporal distribution
INDATA.mask = np.zeros(50)
INDATA.mask[25] = True

OCDATA = np.ma.zeros(5) # this makes setting the expected flags easy
OCDATA.mask = np.zeros(5)
OCDATA.mask[2] = True


def _generate_pd_times(length: int, start_dt: dt.datetime) -> pd.DataFrame:
    """generate a pandas dataframe of the times

    Parameters
    ----------
    length : int
        length of the time dataframe to be generated
    start_dt: dt.datetime
        start datetime of the dataframe

    Returns
    -------
    pd.DataFrame
        DataFrame of datetimes
    """

    return pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                          for i in range(length)])[0])


def _generate_expected_flags(data: np.ndarray) -> np.ndarray:
    """ Generate the expected flags
    Use the fact that simulated odd cluster data is zeros, but
    non cluster data are ones

    Parameters
    ----------
    data : np.ndarray
        length of the time dataframe to be generated

    Returns
    -------
    np.ndarray
        array of flag values (strings)
    """
    expected_flags = np.array(["" for _ in data])
    expected_flags[data == 0] = "o"

    return expected_flags


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


@patch("odd_cluster.logger")
def test_flag_clusters_none(logger_mock: Mock) -> None:
    """Ensure logger called even if no clusters were flagged"""
    temperature = common.example_test_variable("temperature", INDATA)

    # make Station
    station = common.example_test_station(temperature)

    odd_cluster.flag_clusters(temperature, station)

    # if no clusters, then just call logger twice and exit
    assert logger_mock.info.call_count == 2


def test_flag_clusters_start() -> None:
    """Mock up a cluster of isolated data at the start of a run"""
    # cluster (length 5), then standard data
    oc_dt = dt.datetime(2000, 4, 1, 0, 0)
    oc_times = _generate_pd_times(len(OCDATA), oc_dt)
    # separation of two months
    end_dt = dt.datetime(2000, 6, 1, 0, 0)
    end_times = _generate_pd_times(len(INDATA), end_dt)

    all_times = pd.concat([oc_times, end_times])
    all_data = np.append(OCDATA, INDATA)

    temperature = common.example_test_variable("temperature", all_data)

    # make Station
    station = common.example_test_station(temperature, times=all_times)

    odd_cluster.flag_clusters(temperature, station)

    expected_flags = _generate_expected_flags(all_data)

    np.testing.assert_array_equal(expected_flags, temperature.flags)


def test_flag_clusters_end() -> None:
    """Mock up a cluster of isolated data at the end of a run"""

    # standard data, then cluster (length 5)
    start_dt = dt.datetime(2000, 4, 1, 0, 0)
    start_times = _generate_pd_times(len(INDATA), start_dt)
    # separation of 2 months
    oc_dt = dt.datetime(2000, 6, 1, 0, 0)
    oc_times = _generate_pd_times(len(OCDATA), oc_dt)

    all_times = pd.concat([start_times, oc_times])
    all_data = np.append(INDATA, OCDATA)

    temperature = common.example_test_variable("temperature", all_data)

    # make Station
    station = common.example_test_station(temperature, times=all_times)

    odd_cluster.flag_clusters(temperature, station)

    expected_flags = _generate_expected_flags(all_data)

    np.testing.assert_array_equal(expected_flags, temperature.flags)


def test_flag_clusters_normal() -> None:
    """Mock up a cluster of isolated data in the middle of a run"""

    start_dt = dt.datetime(2000, 2, 1, 0, 0)
    start_times = _generate_pd_times(len(INDATA), start_dt)
    # separated by two months, cluster length of 5
    oc_dt = dt.datetime(2000, 4, 1, 0, 0)
    oc_times = _generate_pd_times(len(OCDATA), oc_dt)
    # separated by two months
    end_dt = dt.datetime(2000, 6, 1, 0, 0)
    end_times = _generate_pd_times(len(INDATA), end_dt)

    all_times = pd.concat([start_times, oc_times, end_times])
    all_data = np.append(INDATA, OCDATA)
    all_data = np.append(all_data, INDATA)

    temperature = common.example_test_variable("temperature", all_data)

    # make Station
    station = common.example_test_station(temperature, times=all_times)

    odd_cluster.flag_clusters(temperature, station)

    expected_flags = _generate_expected_flags(all_data)

    np.testing.assert_array_equal(expected_flags, temperature.flags)


@patch("odd_cluster.flag_clusters")
def test_read_hcc(flag_clusters_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    odd_cluster.occ(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    flag_clusters_mock.assert_called_once()
