"""
Contains tests for qc_utils.py
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import qc_utils
import common


def test_gcv_zeros_in_central_section_nozeros() -> None:

    histogram = np.arange(1, 15, 1)
    n_zeros = qc_utils.gcv_zeros_in_central_section(histogram, 10)

    assert n_zeros == 0


@pytest.mark.parametrize("length, inner_n", [(5, 5), (10, 10)])
def test_gcv_zeros_in_central_section(length: int,
                                      inner_n: int) -> None:

    histogram = np.arange(0, length, 1)
    histogram[:4] = 0

    n_zeros = qc_utils.gcv_zeros_in_central_section(histogram, inner_n)

    assert n_zeros == 4


def test_gcv_linear_fit_to_log_histogram() -> None:

    histogram = np.array([1000, 333, 100, 33, 10, 3.3, 1])
    bins = np.arange(histogram.shape[0])

    result = qc_utils.gcv_linear_fit_to_log_histogram(histogram, bins)

    np.testing.assert_array_almost_equal(np.array([3, -0.5]), result, decimal=2)


def test_get_critical_values_identical() -> None:

    indata = np.ones(100)
    threshold = qc_utils.get_critical_values(indata)

    assert threshold == 2


def test_get_critical_values_none() -> None:

    indata = np.array([])
    binwidth = 10

    threshold = qc_utils.get_critical_values(indata, binwidth=binwidth)

    assert threshold == binwidth


@pytest.mark.parametrize("length, n_zeros", [(5, 3), (10, 7)])
def test_get_critical_values_many_zeros(length: int,
                                        n_zeros: int) -> None:
    # testing both short (5) and longer (10) arrays with
    #   sufficient zeros to trigger an exit
    indata = np.arange(length)
    indata[:n_zeros] = 0
    binwidth = 10

    threshold = qc_utils.get_critical_values(indata, binwidth=binwidth)

    assert threshold == binwidth + indata[-1]


@patch("qc_utils.np.histogram")
def test_get_critical_values_positive_slope(histogram_mock: Mock) -> None:
    # Mocking histogram values for use in log_space
    #   Can't have streaks of length 0 or 1, so these set to 0
    hist = np.array([0, 0, 100, 100, 200, 200, 300, 300, 300, 350, 50, 0, 0, 1])
    bins = np.arange(len(hist) + 1)
    histogram_mock.return_value = (hist, bins)

    threshold = qc_utils.get_critical_values(bins)
    # if positive slope, then threshold is max(indata) + binwidth
    #    As histogram mocked, indata only used for this default calculation
    #    Passing in "bins", so tested result as indicated.
    assert threshold == max(bins) + 1


@patch("qc_utils.np.histogram")
def test_get_critical_values_no_non_zero(histogram_mock: Mock) -> None:
    # Mocking histogram values for use in log_space
    #   Can't have streaks of length 0 or 1, so these set to 0
    hist = np.array([0, 0, 1000, 333, 100, 33, 10, 3.3, 1, 1, 1, 1, 1, 1])
    bins = np.arange(len(hist) + 1)
    histogram_mock.return_value = (hist, bins)

    threshold = qc_utils.get_critical_values(bins)
    # if no zero bins after fit crosses 0.1, then threshold is max(indata) + binwidth
    #    As histogram mocked, indata only used for this default calculation
    #    Passing in "bins", so tested result as indicated.
    assert threshold == max(bins) + 1


@patch("qc_utils.np.histogram")
def test_get_critical_values_normal(histogram_mock: Mock) -> None:

    # Create values for all space
    fit = np.array([10000, 3333, 1000, 333, 100, 33, 10, 3.3, 1, 0.33, 0.1, 0.033, 0.01, 0.003])
    hist = np.copy(fit)
    # Can't have streaks of length 0 or 1, so these set to 0
    hist[:2] = 0
    # And remove high values where instances < 1
    hist[hist<1] = 0

    bins = np.arange(len(hist) + 1)
    histogram_mock.return_value = (hist, bins)

    threshold = qc_utils.get_critical_values(np.arange(10))#, plots=True)
    # checked via plots
    assert threshold == np.nonzero(fit < 0.1)[0][0]


def test_prepare_data_repeating_streak_indices() -> None:

    # locations in an array where DPD = 0
    locs = np.array([0,
                     10, 11, 12, 13, 14, 15,
                     20, 21, 22, 23,
                     30, 31, 32, 33,
                     40, 41, 42,
                     50, 51, 52,
                     60, 61, 62,
                     70])

    # diff=1 to test that locations of DPD=0 are neighbouring
    lengths, grouped_diffs, streaks = qc_utils.prepare_data_repeating_streak(locs, diff=1)

    # lengths which are passed into the fitting to
    np.testing.assert_array_equal(lengths, np.array([6, 4, 4, 3, 3, 3]))
    # grouped first differences
    np.testing.assert_array_equal(grouped_diffs, np.array([[10,  1],
                                                           [ 1,  5],
                                                           [ 5,  1],
                                                           [ 1,  3],
                                                           [ 7,  1],
                                                           [ 1,  3],
                                                           [ 7,  1],
                                                           [ 1,  2],
                                                           [ 8,  1],
                                                           [ 1,  2],
                                                           [ 8,  1],
                                                           [ 1,  2],
                                                           [ 8,  1]]))
    # locations in the grouped differences which are repeated streaks
    np.testing.assert_array_equal(streaks, np.array([1, 3, 5, 7, 9, 11]))

    assert len(lengths) == len(streaks)


def test_prepare_data_repeating_streak_values() -> None:

    # array containing streaks
    inarray = np.ma.arange(0, 200, 1)
    inarray.mask = np.zeros(inarray.shape[0])

    # make some streaks (set start index and length)
    common.generate_streaky_data(inarray, common.REPEATED_STREAK_STARTS_LENGTHS)

    # diff=0 for neighbouring values being identical
    lengths, _, streaks = qc_utils.prepare_data_repeating_streak(inarray, diff=0)

    # lengths which are passed into the fitting to
    np.testing.assert_array_equal(lengths, np.fromiter(common.REPEATED_STREAK_STARTS_LENGTHS.values(),
                                                       dtype=int))

    # locations in the grouped differences which are repeated streaks
    np.testing.assert_array_equal(streaks,
                                  np.array([ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]))

    assert len(lengths) == len(streaks)


def test_gcv_calculate_binmax() -> None:

    indata = np.arange(10)
    binmin = 0
    binwidth = 0.1

    binmax = qc_utils.gcv_calculate_binmax(indata, binmin, binwidth)

    assert binmax == 18

def test_gcv_calculate_binmax_large() -> None:

    indata = np.arange(10)
    indata[-1] = 2001
    binmin = 0
    binwidth = 0.1

    binmax = qc_utils.gcv_calculate_binmax(indata, binmin, binwidth)

    assert binmax == 2000


def test_update_dataframe() -> None:

    data = {"Year" : [2020, 2020, 2020, 2020, 2020],
            "Month" : [1, 1, 1, 1, 1],
            "Day" : [10, 11, 12, 13, 14],
            "wind_direction" : [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)

    indata = np.array([100, 200, 300, 400, 500])

    locations = np.array([False, False, True, False, False])

    column = "wind_direction"

    qc_utils.update_dataframe(df, indata, locations, column)

    expected = np.array([10, 20, 300, 40, 50])
    np.testing.assert_array_equal(df["wind_direction"].to_numpy(),
                                  expected)
   

def test_create_bins() -> None:
    """Simple test of bin creation"""

    indata = np.array([1, 10])

    result = qc_utils.create_bins(indata, 0.5, "dummy")

    expected = np.arange(1-2.5, 10+2.5, 0.5)

    np.testing.assert_array_equal(result, expected)


def test_create_bins_long() -> None:
    """Simple test of bin creation when max and min would result in too many"""

    indata = np.array([-7000, 7000])

    result = qc_utils.create_bins(indata, 0.5, "temperature")

    # -89.2 to 56.7
    expected = np.arange(-190-2.5, 157+2.5, 0.5)

    np.testing.assert_array_equal(result, expected)
