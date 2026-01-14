"""
Contains tests for utils.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
import pytest
from unittest.mock import patch

import utils
import setup

# testing classes

def test_metvar() -> None:
    """Test Met Var has correct attribute values"""
    obsvar = utils.MeteorologicalVariable("Name", -99.9, "hPa", "float")

    assert obsvar.name == "Name"
    assert obsvar.mdi == -99.9
    assert obsvar.units == "hPa"
    assert obsvar.dtype == "float"


def test_metvar_data() -> None:
    """Test Met Var stores data in correct place and way"""
    obsvar = utils.MeteorologicalVariable("Name", -99.9, "hPa", "float")
    obsvar.store_data(np.ma.arange(5))

    np.testing.assert_array_equal(obsvar.data, np.ma.arange(5))
    assert isinstance(obsvar.data, np.ma.MaskedArray)


def test_metvar_flags() -> None:
    """Test Met Var stores flags in correct place and way"""
    obsvar = utils.MeteorologicalVariable("Name", -99.9, "hPa", "float")
    inflags = np.array(["", "t", "e", "s", "t"])
    obsvar.store_flags(inflags)

    np.testing.assert_array_equal(obsvar.flags, inflags)


def test_station() -> None:
    """Test Station has correct attributes"""
    station = utils.Station("ID", 90., 180., 100.)

    assert station.id == "ID"
    assert station.lat == 90.
    assert station.lon == 180.
    assert station.elev == 100.

    # and empty defaults
    assert station.country == ""
    assert station.continent == ""

    for obs_var in setup.obs_var_list:
        assert hasattr(station, obs_var)


def test_station_times() -> None:
    """Test Station stores times in correct way"""
    station = utils.Station("ID", 90., 180., 100.)

    times = pd.Series([dt.datetime(2000, 1, 1, 12, 45)])
    station.set_times(times)

    pd.testing.assert_series_equal(station.times, times)


def test_station_datetimes() -> None:
    """Test Station stores datetime parameters in correct way"""
    station = utils.Station("ID", 90., 180., 100.)
    station.set_datetime_values(np.array([2000]), np.array([1]),
                                np.array([1]), np.array([12]))

    np.testing.assert_array_equal(station.years, np.array([2000]))
    np.testing.assert_array_equal(station.months, np.array([1]))
    np.testing.assert_array_equal(station.days, np.array([1]))
    np.testing.assert_array_equal(station.hours, np.array([12]))


# testing routines

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


example_stn_list = Path(__file__).parent / "example_data/station_list_fwf.txt"
@patch("utils.setup.STATION_LIST", example_stn_list)
def test_get_station_list_fwf_endid() -> None:
    """Test reading of fixed-width format station list with end_id"""

    df = utils.get_station_list(end_id="USW00094849")

    expected_columns = np.array(['id', 'latitude', 'longitude',
                                 'elevation', 'state', 'name', 'wmo'])
    np.testing.assert_array_equal(df.columns,
                                  expected_columns)
    assert df.shape == (3, 7)
    assert df["id"].iloc[-1] == "USW00094849"
    assert df["name"].iloc[-1] == "ALPENA CO RGNL AP"
    assert df["wmo"].iloc[-1] == 72639


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


example_stn_list = Path(__file__).parent / "example_data/station_list_csv.txt"
@patch("utils.setup.STATION_LIST", example_stn_list)
def test_get_station_list_csv_restart_id() -> None:
    """Test reading of "csv" format station list"""

    df = utils.get_station_list(restart_id="USW00094847")

    expected_columns = np.array(['id', 'latitude', 'longitude',
                                 'elevation', 'state', 'name', 'wmo'])
    np.testing.assert_array_equal(df.columns,
                                  expected_columns)
    assert df.shape == (3, 7)
    assert df["id"].iloc[0] == "USW00094847"
    assert df["name"].iloc[0] == "DETROIT METRO AP"
    assert df["wmo"].iloc[0] == ""


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


def test_insert_flags() -> None:
    """Test that flag insertion works"""

    flags1 = np.array(["l" for i in range(10)])
    flags2 = np.array(["a" for i in range(10)])
    expected = np.array(["la" for i in range(10)])

    result = utils.insert_flags(flags1, flags2)

    np.testing.assert_array_equal(result, expected)


def test_insert_flags_empty() -> None:
    """Test that flag insertion works"""

    flags1 = np.array(["l" for i in range(10)])
    flags2 = np.array(["a" for i in range(10)])
    flags1[0] = ""
    flags2[1] = ""
    expected = np.array(["la" for i in range(10)])
    expected[:2] = ["a", "l"]

    result = utils.insert_flags(flags1, flags2)

    np.testing.assert_array_equal(result, expected)
