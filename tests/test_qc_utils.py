"""
Contains tests for qc_utils.py
"""
import numpy as np

import qc_utils


def test_prepare_data_repeating_string_indices():

    # locations in an array where DPD = 0
    locs = np.array([0,
                     10, 11, 12, 13, 14, 15,
                     20, 21, 22, 23,
                     30, 31, 32, 33,
                     40, 41, 42,
                     50, 51, 52,
                     60, 61, 62,
                     70])

    lengths, grouped_diffs, strings = qc_utils.prepare_data_repeating_string(locs, diff=1)

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
    # locations in the grouped differences which are repeated strings
    np.testing.assert_array_equal(strings, np.array([1, 3, 5, 7, 9, 11]))

    assert len(lengths) == len(strings)

#def test_prepare_data_repeating_string_values():

    # as above but with diff=0

def test_gcv_calculate_binmax():

    indata = np.arange(10)
    binmin = 0
    binwidth = 0.1

    binmax = qc_utils.gcv_calculate_binmax(indata, binmin, binwidth)

    assert binmax == 18

def test_gcv_calculate_binmax_large():

    indata = np.arange(10)
    indata[-1] = 2001
    binmin = 0
    binwidth = 0.1

    binmax = qc_utils.gcv_calculate_binmax(indata, binmin, binwidth)

    assert binmax == 2000

# def test_gcv_central_section():
    