"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock

import distribution_all

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


def test_find_monthly_scaling() -> None:
    """Test calculation and storage of scaling parameters"""

    indata = np.ma.array(np.arange(200))
    config_dict = {}

    result = distribution_all.find_monthly_scaling(indata, config_dict,
                                                   "temperature", 1)

    assert result[0] == qc_utils.average(indata)
    assert result[1] == qc_utils.spread(indata)

    assert config_dict["ADISTRIBUTION-temperature"]["1-clim"] == result[0]
    assert config_dict["ADISTRIBUTION-temperature"]["1-spread"] == result[1]


def test_find_monthly_scaling_no_data() -> None:
    """Test calculation and storage of scaling parameters"""

    indata = np.ma.array(np.arange(10))
    config_dict = {}

    result = distribution_all.find_monthly_scaling(indata, config_dict,
                                                   "temperature", 1)

    assert result[0] == utils.MDI
    assert result[1] == utils.MDI

    assert config_dict["ADISTRIBUTION-temperature"]["1-clim"] == result[0]
    assert config_dict["ADISTRIBUTION-temperature"]["1-spread"] == result[1]


@patch("distribution_all.find_monthly_scaling")
def test_prepare_all_data_full_calls_mdi(find_monthly_mock: Mock) -> None:
    """Test when doing a full run, and with MDI as returns"""

    find_monthly_mock.return_value = (utils.MDI, utils.MDI)

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               {}, full=True)

    # test calls
    find_monthly_mock.assert_called_once()
    calls = find_monthly_mock.call_args_list[0]
    np.testing.assert_array_equal(calls.args[0],
                                  station.temperature.data)
    assert calls.args[1] == {}
    assert calls.args[2] == "temperature"
    assert calls.args[3] == 1

    # test result
    np.testing.assert_almost_equal(result, np.ma.array([utils.MDI]))


def test_prepare_all_data_spread_zero() -> None:
    """Test when reading from dict, with spread of zero"""

    temperature_values = [("1-clim", 1)]
    temperature_values += [("1-spread", 0)]
    config_dict = {"ADISTRIBUTION-temperature": dict(temperature_values)}

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               config_dict, full=False)

    # test result
    np.testing.assert_almost_equal(result,
                                   np.ma.arange(10)-1)


def test_prepare_all_data_spread_nonzero() -> None:
    """Test when reading from dict, with nonzero spread"""

    temperature_values = [("1-clim", 1)]
    temperature_values += [("1-spread", 2)]
    config_dict = {"ADISTRIBUTION-temperature": dict(temperature_values)}

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               config_dict, full=False)

    # test result
    np.testing.assert_almost_equal(result,
                                   (np.ma.arange(10)-1)/2)



@patch("distribution_all.all_obs_gap")
def test_dgc(all_gap_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    distribution_all.dgc(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    all_gap_mock.assert_called_once()