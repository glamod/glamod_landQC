"""
Contains tests for odd_cluster.py
"""
import numpy as np
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
def test_find_monthly_scaling_small_spread(prepare_mock: Mock) -> None:
    """writing of config dictionary if data has small spread"""
    prepare_mock.return_value = np.ma.array([2, 2, 2, 3, 3, 3])
    station = _setup_station(np.ma.arange(10))
    config_dict = {}

    # Do the call
    distribution_monthly.find_monthly_scaling(
        station.temperature, station, config_dict)

    # Mock to check call occurs as expected with right return
    assert config_dict["MDISTRIBUTION-temperature"]["1-clim"] == 2.5
    assert config_dict["MDISTRIBUTION-temperature"]["1-spread"] == 2


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


def test_flag_large_offsets() -> None:
    """Test that large offsets are flagged"""
    # 10 years of data for January
    station = common.generate_station_for_clim_and_dist_tests()
    flags = np.array(["" for _ in station.temperature.data])

    standard_months = np.array([6, 5, 4, 3, 0, 0, -3, -4, -5, -6])

    distribution_monthly.flag_large_offsets(station, 1, standard_months, flags)

    expected = np.array(["" for _ in station.temperature.data])
    # first two and last two months should be flagged
    expected[:24*31*2] = "D"
    expected[-24*31*2:] = "D"

    np.testing.assert_array_equal(flags, expected)


def test_walk_distribution_all_same() -> None:
    """Test that walking of distribution results in no flags if all
    standardised_months are zero"""

    standard_months = np.array([0 for _ in range(10)])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set
    expected = []

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_some_zero() -> None:
    """Test that walking of distribution results in no flags if all
    standardised_months are zero"""
    standard_months = np.array([0. for _ in range(10)])
    standard_months[:4] = [0.1, 0.2, 0.15, 0.05]

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as one of pair always zero
    expected = []

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_odd_end_of_branch() -> None:
    """Test that walking of distribution results in no flags if
    values are not identical, or zero, but sufficiently symmetrical,
    so that it reaches the end of the distribution arms, with no flags set
    (odd number of months)"""
    # symmetrical, but not identifical
    standard_months = np.array([-0.55, -0.45, -0.35, -0.25, -0.15, 0,
                                0.1, 0.2, 0.3, 0.4, 0.5])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, pair are close (though not identical)
    expected = []

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_even_end_of_branch() -> None:
    """Test that walking of distribution results in no flags if
    values are not identical, or zero, but sufficiently symmetrical,
    so that it reaches the end of the distribution arms, with no flags set
    (even number of months)"""
    # symmetrical, but not identifical
    standard_months = np.array([-0.55, -0.45, -0.35, -0.25, -0.15,
                                0.1, 0.2, 0.3, 0.4, 0.5])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, pair are close (though not identical)
    expected = []

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_odd_upper_long_min_small() -> None:
    """Test that walking of distribution results in no flags even
    if asymmetric, when min of pair is too close to zero
    (odd number of months)"""

    standard_months = np.array([-0.55, -0.45, -0.35, -0.25, -0.15, 0,
                                0.1, 0.2, 0.3, 1.0, 2.0])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as min of assymetric pair too small
    expected = []

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_odd_upper_long() -> None:
    """Test that walking of distribution results in expected flags
    for upper tail (odd number of months)"""

    standard_months = np.array([-1.55, -0.45, -0.35, -0.25, -0.15, 0,
                                0.1, 0.2, 0.3, 1.0, 4.0])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as min of assymetric pair too small
    expected = [10]

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_even_upper_long() -> None:
    """Test that walking of distribution results in expected flags
    for upper tail (even number of months)"""

    standard_months = np.array([-1.55, -0.45, -0.35, -0.25, -0.15, 0,
                                0.1, 0.2, 0.3, 1.0, 4.0])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as min of assymetric pair too small
    expected = [10]

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_odd_lower_long() -> None:
    """Test that walking of distribution results in expected flags
    for lower tail (odd number of months)"""

    standard_months = np.array([-4, -3, -0.35, -0.25, -0.15, 0,
                                0.1, 0.2, 0.3, 1.4, 1.6])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as min of assymetric pair too small
    expected = [0]

    np.testing.assert_array_equal(result, expected)


def test_walk_distribution_even_lower_long() -> None:
    """Test that walking of distribution results in expected flags
    for lower tail (even number of months)"""

    standard_months = np.array([-4, -3, -0.35, -0.25, -0.15,
                                0.1, 0.2, 0.3, 1.4, 1.6])

    result = distribution_monthly.walk_distribution(standard_months)

    # no flags set, as min of assymetric pair too small
    expected = [0]

    np.testing.assert_array_equal(result, expected)


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


@patch("distribution_monthly.walk_distribution")
@patch("distribution_monthly.flag_large_offsets")
@patch("distribution_monthly.prepare_monthly_data")
def test_monthly_gap(prepare_mock: Mock,
                     flag_large_mock: Mock,
                     walk_mock: Mock) -> None:
    """Test gap finding routine logic and calls to other routines"""
    prepare_mock.return_value = np.ma.array([1, 2, 3, 4, 5])
    station = _setup_station(np.ma.arange(10))

    walk_mock.return_value = [0] # which month to flag (all values given mocked station)

    # build up config_dict, but values only for January
    temperature_values = [("1-clim", 1)]
    temperature_values += [(f"{month}-clim", utils.MDI) for month in range(2, 13)]
    temperature_values += [("1-spread", 2)]
    temperature_values += [(f"{month}-spread",   utils.MDI) for month in range(2, 13)]
    config_dict = {"MDISTRIBUTION-temperature": dict(temperature_values)}

    # Do the call
    distribution_monthly.monthly_gap(
        station.temperature, station, config_dict)

    # expected values
    standard_months = np.ma.array([0, 0.5, 1, 1.5, 2])
    expected_flags = np.array(["D" for i in range(10)])

    # check call values for this routine
    flag_large_mock.assert_called_once()  # only values for January
    calls = flag_large_mock.call_args_list[0]
    assert calls.args[0] == station
    assert calls.args[1] == 1
    np.testing.assert_array_equal(standard_months, calls.args[2])
    np.testing.assert_array_equal(expected_flags, calls.args[3])

    # and this routine
    walk_mock.assert_called_once()  # only values for January
    calls = walk_mock.call_args_list[0]
    np.testing.assert_array_equal(standard_months, calls.args[0])

    # finally check that flags set as expected (all values given mocked inputs)
    np.testing.assert_array_equal(station.temperature.flags,
                                  expected_flags)



@patch("distribution_monthly.find_monthly_scaling")
@patch("distribution_monthly.monthly_gap")
def test_dgc(monthly_gap_mock: Mock,
             scaling_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    distribution_monthly.dgc(station, ["temperature"], {}, full=True)

    # Mock to check call occurs as expected with right return
    monthly_gap_mock.assert_called_once()
    scaling_mock.assert_called_once()
