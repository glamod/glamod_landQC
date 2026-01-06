"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock, call

import distribution_monthly

import common
import utils
import qc_utils

def _setup_station(indata: np.ma.MaskedArray) -> utils.Station:
    """Create a station object to hold the information enabling
    the QC test to be tested

    Parameters
    ----------
    indata : np.ma.MaskedArray
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


def test_prepare_monthly_data() -> None:
    """Check calculation of monthly averages"""

    station = _setup_station(np.ma.ones(31*24))

    result = distribution_monthly.prepare_monthly_data(
        station.temperature,
        station, 1
        )

    np.testing.assert_array_equal(result,
                                  np.ma.array([1.0]))


def test_prepare_monthly_data_none() -> None:
    """Check calculation of monthly averages if no data"""

    station = _setup_station(np.ma.ones(28))

    result = distribution_monthly.prepare_monthly_data(
        station.temperature,
        station, 1
        )
    expected = np.ma.array([0.0], mask=True)

    np.testing.assert_array_equal(result,
                                  expected)


@patch("distribution_monthly.prepare_monthly_data")
def test_find_monthly_scaling_few_months(prepare_mock: Mock) -> None:
    """writing of config dictionary if insufficient months of data"""
    prepare_mock.return_value = np.ma.array([1, 2, 3, 4])
    station = _setup_station(np.ma.arange(10))
    config_dict = {}

    # Do the call
    distribution_monthly.find_monthly_scaling(
        station.temperature, station, config_dict)

    # Mock to check call occurs as expected with right return
    assert config_dict["MDISTRIBUTION-temperature"]["1-clim"] == utils.MDI
    assert config_dict["MDISTRIBUTION-temperature"]["1-spread"] == utils.MDI


@patch("distribution_monthly.prepare_monthly_data")
def test_find_monthly_scaling(prepare_mock: Mock) -> None:
    """writing of config dictionary"""
    prepare_mock.return_value = np.ma.array([1, 2, 3, 4, 5])
    station = _setup_station(np.ma.arange(10))
    config_dict = {}

    # Do the call
    distribution_monthly.find_monthly_scaling(
        station.temperature, station, config_dict)

    # Mock to check call occurs as expected with right return
    assert config_dict["MDISTRIBUTION-temperature"]["1-clim"] == 3
    assert config_dict["MDISTRIBUTION-temperature"]["1-spread"] == qc_utils.spread(np.arange(1, 6))


@patch("distribution_monthly.prepare_monthly_data")
def test_monthly_gap_no_clim_no_spread(prepare_mock: Mock) -> None:
    """Test gap finding if no climatology or spread available"""
    prepare_mock.return_value = np.ma.array([1, 2, 3, 4, 5])
    station = _setup_station(np.ma.arange(10))

    # build up config_dict
    temperature_values = [(f"{month}-clim", utils.MDI) for month in range(1, 13)]
    temperature_values += [(f"{month}-spread", utils.MDI) for month in range(1, 13)]
    config_dict = {"MDISTRIBUTION-temperature": dict(temperature_values)}

    # Do the call
    distribution_monthly.monthly_gap(
        station.temperature, station, config_dict)

    calls = [call(station.temperature, station,
                  month, diagnostics=False) for month in range(1, 13)]

    prepare_mock.assert_has_calls(calls)


def test_flag_large_offsets() -> None:
    """Test that large offsets are flagged"""
    # 10 years of data for January
    station = common.generate_station_for_clim_and_dist_tests()
    flags = np.array(["" for _ in station.temperature.data])

    standard_months = [6, 5, 4, 3, 0, 0, -3, -4, -5, -6]

    distribution_monthly.flag_large_offsets(station, 1, standard_months, flags)

    expected = np.array(["" for _ in station.temperature.data])
    # first two and last two months should be flagged
    expected[:24*31*2] = "D"
    expected[-24*31*2:] = "D"

    np.testing.assert_array_equal(flags, expected)





@patch("distribution_monthly.monthly_gap")
def test_dgc(monthly_gap_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    distribution_monthly.dgc(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    monthly_gap_mock.assert_called_once()
