"""
Contains tests for common.py
"""
import numpy as np
import qc_utils as utils

import common

def test_example_test_variable() -> None:

    testdata = np.arange(5)
    result = common.example_test_variable("test",
                                          testdata)

    assert isinstance(result, utils.Meteorological_Variable)
    np.testing.assert_array_equal(result.data, testdata)


def test_example_test_station() -> None:

    testdata = np.arange(5)
    this_var = common.example_test_variable("test",testdata)
    result = common.example_test_station(this_var)

    assert isinstance(result, utils.Station)
    assert hasattr(result, "test")
    assert result.lat == 45


def test_generate_streaky_data() -> None:

    testdata = np.arange(10)
    testdict = {5: 3}
    expected = np.arange(10)
    expected[5:8] = expected[5]

    streaky_data = common.generate_streaky_data(testdata, testdict)

    np.testing.assert_array_equal(testdata, expected)
