"""
Contains tests for diurnal.py
"""
import numpy as np
import pytest
from unittest.mock import call, patch, Mock

import diurnal

import common
import qc_utils as utils

def test_make_sines() -> None:
    """Test that stacked sine curves created correctly"""
    sines = diurnal.make_sines()

    expected = np.array([0.5, 0.37059048, 0.25, 0.14644661, 0.0669873, 0.01703709,
                         0., 0.01703709, 0.0669873, 0.14644661, 0.25, 0.37059048,
                         0.5, 0.62940952, 0.75, 0.85355339, 0.9330127, 0.98296291,
                         1., 0.98296291, 0.9330127, 0.85355339, 0.75, 0.62940952])

    # test that sine values for first point in each of the 24
    # rows is as expected
    np.testing.assert_array_almost_equal(sines[:, 0], expected)


@pytest.mark.parametrize("minutes, expected", [(np.array([0, 360, 720, 1080]), True),
                                               (np.array([720, 1080]), False),
                                               (np.array([360, 361, 720]), False),])
def test_quartile_check(minutes: np.ndarray,
                        expected: bool) -> None:
    """Test that quartile check returns True if obs in each 3hr quartile"""

    result = diurnal.quartile_check(minutes)

    assert result == expected


def test_make_scaled_sine() -> None:
    """Test that scaling of sine curve occurs as expected"""
    input_sine = diurnal.make_sines()

    maximum = 10
    minimum = -10

    expected = (input_sine * 20) + minimum

    result = diurnal.make_scaled_sine(maximum, minimum)

    assert np.max(result) == maximum
    assert np.min(result) == minimum

    np.testing.assert_array_equal(result, expected)


def test_find_differences() -> None:
    """Test phase differences between sine curves found sensibly"""

    # example data is in total phase with constructed sines
    day_data = np.array([0, 1, 0, -1])
    minutes = np.array([0, 360, 720, 1080])

    result = diurnal.find_differences(day_data, minutes)
    print(result)
    # test first entry is minimum and close to zero
    assert np.min(result) == result[0]
    np.testing.assert_almost_equal(np.min(result), 0)


def test_find_uncertainties() -> None:
    """Test uncertainties are found as expected"""

    differences = np.array([7,7,7,7,7,6,
                            5,4,3,2,1,0,
                            1,1.5,2,3,4,5,
                            6,7,7,7,7,7])

    # hence values under threshold (<3) should be [2,1,0,1,1.5,2]
    #  indices [ 9 10 11 12 13 14]
    # best fit == 11
    # uncertainty dominated by RHS, 14-11+1 results in 4.

    best_fit = np.argmin(differences)

    result = diurnal.find_uncertainty(differences, best_fit)

    assert result == 4


def test_find_fit_and_uncertainty() -> None:
    """Testing calls to calculation routines work as expected"""

    # example data is in total phase with constructed sines
    day_data = np.array([0, 1, 0, -1])
    minutes = np.array([0, 360, 720, 1080])

    fit, unc = diurnal.find_fit_and_uncertainty(day_data, minutes)

    # in total phase with constructed sine
    assert fit == 0
    # uncertainty tested elsewhere


def _setup_station(indata: np.ma.array) -> utils.Station:

    # set up the data
    indata.mask = np.zeros(len(indata))

    # make MetVars
    temperature = common.example_test_variable("temperature", indata)

    # make Station
    station = common.example_test_station(temperature)

    return station

@patch("diurnal.find_fit_and_uncertainty")
def test_get_daily_offset(find_fit_mock: Mock) -> None:
    """Test calculation of fit/uncertainty from station object"""

    find_fit_mock.return_value = (0, 0)

    # set up some random data
    dummy_station = _setup_station(np.ma.arange(24))
    obs_var = dummy_station.temperature

    locs = np.arange(24)

    # no need to test returns, these done elsewhere
    _, _ = diurnal.get_daily_offset(dummy_station, locs, obs_var)

    find_fit_mock.assert_called_once()

    expected_data = np.ma.arange(24)  # data for first 24hr
    expected_data.mask = np.zeros(len(expected_data))
    expected_mins = np.arange(0, 1440, 60)  # minutes

    # testing calls
    np.testing.assert_array_equal(find_fit_mock.call_args[0][0], expected_data)
    np.testing.assert_array_equal(find_fit_mock.call_args[0][1], expected_mins)


