"""
Contains tests for variance.py
"""
import numpy as np
import pandas as pd
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import variance
import common
import utils
import qc_utils


def _setup_station() -> utils.Station:
    """Create a station with temperatures for 10 Januaries

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
    temperature = common.example_test_variable("temperature", indata)

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
    station = common.example_test_station(temperature, times)

    return station


def test_calculate_climatology() -> None:
    """Test climatology"""

    # 5 times 1 to 100
    indata = np.ma.MaskedArray(np.repeat(np.arange(100) + 1, 5))

    clim, c_mask = variance.calculate_climatology(indata,
                                                  winsorize=False)

    assert not c_mask  # test if False
    assert clim == 50.5


def test_calculate_climatology_winsorize() -> None:
    """Test climatology with winsorizing option"""

    # 5 times 1 to 100
    indata = np.ma.MaskedArray(np.repeat(np.arange(100) + 1, 5))

    clim, c_mask = variance.calculate_climatology(indata,
                                                  winsorize=True)

    assert not c_mask  # test if False
    assert clim == 50.45


def test_calculate_hourly_anomalies() -> None:

    # station has values of 0..23 repeating for each hour
    station = _setup_station()

    result = variance.calculate_hourly_anomalies(station.hours,
                                                 station.temperature.data)

    # hence anomalies will all be zero
    assert np.all(result == 0)


def test_calculate_hourly_anomalies_nonzero() -> None:

    # station has values of 0..23 repeating for each hour
    station = _setup_station()

    # offset first and last day
    station.temperature.data[:24] += 1
    station.temperature.data[-24:] -= 1

    result = variance.calculate_hourly_anomalies(station.hours,
                                                 station.temperature.data)

    # check first and last days, as well as the rest are as expected
    assert np.all(result[24: -24] == 0)
    assert np.all(result[:24] == 1)
    assert np.all(result[-24:] == -1)


def test_normalise_hourly_anomalies() -> None:

    anomalies = np.ma.arange(20)

    result = variance.normalise_hourly_anomalies(anomalies)

    spread = qc_utils.spread(anomalies)

    np.testing.assert_allclose(result, anomalies/spread)


def test_normalise_hourly_anomalies_small_spread() -> None:

    anomalies = np.ma.ones(20) * 3

    result = variance.normalise_hourly_anomalies(anomalies)

    np.testing.assert_allclose(result, anomalies/1.5)

@patch("variance.find_thresholds")
@patch("variance.variance_check")
def test_evc(check_mock: Mock,
             find_mock: Mock) -> None:
    """Test driving routine calls functions as expected"""

    temperature = common.example_test_variable("temperature",
                                               np.full(10, 24))

    station = common.example_test_station(temperature)

    variance.evc(station, ["temperature"], {}, full=True)

    find_mock.assert_called_once_with(temperature, station, {},
                                      plots=False, diagnostics=False)
    check_mock.assert_called_once_with(temperature, station, {},
                                       plots=False, diagnostics=False)