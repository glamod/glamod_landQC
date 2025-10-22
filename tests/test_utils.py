"""
Contains tests for utils.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import utils

example_stn_list = Path(__file__).parent / "example_data/station_list_fwf.txt"
@patch("utils.setup.STATION_LIST", example_stn_list)
def test_get_station_list_fwf() -> None:
    """Test reading of fixed-width format station list"""

    df = utils.get_station_list()

    expected_columns = np.array(['id', 'latitude', 'longitude',
                                 'elevation', 'state', 'name', 'wmo'])
    np.testing.assert_array_equal(df.columns,
                                  expected_columns)
    assert df.shape == (4, 7)
    assert df["id"].iloc[0] == "USW00094846"
    assert df["name"].iloc[-1] == "MARQUETTE"
    assert df["wmo"].iloc[-1] == 72743


example_stn_list = Path(__file__).parent / "example_data/station_list_csv.txt"
@patch("utils.setup.STATION_LIST", example_stn_list)
def test_get_station_list_csv() -> None:
    """Test reading of "csv" format station list"""

    df = utils.get_station_list()

    expected_columns = np.array(['id', 'latitude', 'longitude',
                                 'elevation', 'state', 'name', 'wmo'])
    np.testing.assert_array_equal(df.columns,
                                  expected_columns)
    assert df.shape == (4, 7)
    assert df["id"].iloc[0] == "USW00094846"
    assert df["name"].iloc[-1] == "MARQUETTE"
    assert df["wmo"].iloc[-1] == ""


# the mask ("expected") is inverted before application
@pytest.mark.parametrize("ds, expected", [(pd.Series([np.nan, "EXAMPLE", "EX"]),
                                           np.array([True, True, False])),
                                          (pd.Series([np.nan, np.nan, np.nan]),
                                           np.array([True, True, True])),
                                          (pd.Series(["EXAMPLE", "EXAMPLE", "EXAMPLE"]),
                                           np.array([True, True, True])),
                                          (pd.Series(["EX", "EX", "EX"]),
                                           np.array([False, False, False])),
                                          (pd.Series(["nan", np.nan, "EXAMPLE", "EX"]),
                                           np.array([False, True, True, False]))
                                          ])
def test_get_measurement_code_mask(ds: pd.Series,
                                   expected: np.ndarray) -> None:
    """Test that the mask is built correctly from the supplied measurement codes"""

    # Only these strings will be permitted.
    #   "" is interpreted as NaN, and so np.nans are allowed
    test_codes = ["", "EXAMPLE"]
    mask = utils.get_measurement_code_mask(ds, test_codes)

    np.testing.assert_array_equal(mask, expected)


