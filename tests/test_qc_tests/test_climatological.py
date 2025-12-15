"""
Contains tests for climatological.py
"""
import numpy as np
import datetime as dt
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import climatological

import common
import utils
import qc_utils


def _setup_station(varname: str = "temperature") -> utils.Station:
    """Create a station with temperatures (or other metric) for 10 Januaries
    :param str varname: name of variable to create
    Returns
    -------
    utils.Station
        Station object with data in temperature field
    """

    nyears = 10
    month_hours = 24*31
    # 10 years of Januaries
    indata = np.ma.array(np.tile(np.arange(24), 31*nyears))
    indata.mask = np.zeros(len(indata))

    # make MetVars
    obsvar = common.example_test_variable(varname, indata)

    for y in range(nyears):
        start_dt = dt.datetime(2000+y, 1, 1, 0, 0)
        month_times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                                for i in range(month_hours)])[0])
        if y == 0:
            times = month_times.copy()
        else:
            times = pd.concat([times, month_times])
    # check at this generation stage that have all Januaries
    assert np.unique(times.dt.month) == 1

    # make Station, by hand so can set times
    station = common.example_test_station(obsvar, times)

    return station


def test_calculate_climatology() -> None:
    """Test if climatology created correctly"""
    station = _setup_station()

    hmlocs, = np.nonzero(np.logical_and(station.months == 1,
                                        station.hours == 1))

    result = climatological.calculate_climatology(station.temperature,
                                                  hmlocs)

    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_almost_equal(result, np.ma.array(1, mask=False))


def test_calculate_climatology_no_data() -> None:
    """Test if climatology created correctly"""
    station = _setup_station()

    # pass in too few obs for test to run
    hmlocs, = np.nonzero(np.logical_and(station.years == 2000,
                                        station.months == 1,
                                        station.hours == 1))

    result = climatological.calculate_climatology(station.temperature,
                                                  hmlocs)

    assert isinstance(result, np.ma.MaskedArray)
    np.testing.assert_array_almost_equal(result, np.ma.array(0, mask=True))


def test_calculate_anomalies() -> None:
    """Test if anomalies created correctly"""
    station = _setup_station()

    # this will get edited in place
    result = np.ma.zeros(station.temperature.data.shape[0])
    result.mask = np.ones(result.shape[0])  # all masked

    climatological.calculate_anomalies(station,
                                       station.temperature,
                                       result,
                                       1)

    # values are zero because all hours have the same value
    expected = np.ma.zeros(station.temperature.data.shape[0])
    expected.mask = np.zeros(expected.shape[0])  # all unmasked

    np.testing.assert_array_almost_equal(result, expected)


@patch("climatological.calculate_climatology")
def test_calculate_anomalies_nonzero(clim_mock: Mock) -> None:
    """Test if anomalies created correctly, this time with data"""
    station = _setup_station()

    anom_offset = 2
    # mock the climatology, so that anomalies are non-zero
    clim_mock.side_effect = [i - anom_offset for i in np.arange(24)]

    # this will get edited in place
    result = np.ma.zeros(station.temperature.data.shape[0])
    result.mask = np.ones(result.shape[0])  # all masked

    climatological.calculate_anomalies(station,
                                       station.temperature,
                                       result,
                                       1)

    # values are zero because all hours have the same value
    expected = np.ma.ones(station.temperature.data.shape[0]) * anom_offset
    expected.mask = np.zeros(expected.shape[0])  # all unmasked

    np.testing.assert_array_almost_equal(result, expected)


def test_normalise_anomalies() -> None:
    """Test normalising of anomalies occurs correctly"""

    anomalies = np.arange(10)
    mlocs = np.arange(5)

    result = climatological.normalise_anomalies(anomalies, mlocs)

    expected = np.arange(5) / qc_utils.spread(np.arange(5))

    np.testing.assert_almost_equal(result, expected)


def test_calculate_annual_anomalies() -> None:
    """Test calculation of annual anomalies"""
    station = _setup_station()

    result = climatological.calculate_annual_anomalies(station,
                                                        station.temperature)
    assert result.shape[0] == 10

    expected = np.ma.ones(10) * qc_utils.average(np.arange(24))

    np.testing.assert_almost_equal(result, expected)


@patch("climatological.find_month_thresholds")
@patch("climatological.monthly_clim")
def test_dcc(clim_mock: Mock,
             thresholds_mock: Mock) -> None:

    temperatures = common.example_test_variable("temperature",
                                                np.arange(5))
    station = common.example_test_station(temperatures)

    climatological.clim_outlier(station, ["temperature"],
                                {}, full=True)

    clim_mock.assert_called_once()
    thresholds_mock.assert_called_once()

    return