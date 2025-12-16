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


def _setup_station(varname: str = "temperature",
                   nyears: int = 10) -> utils.Station:
    """Create a station with temperatures (or other metric) for 10 Januaries
    :param str varname: name of variable to create

    :param str varname: name to call variable
    :param int nyears: number of years of data to generate

    Returns
    -------
    utils.Station
        Station object with data in temperature field
    """

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


@pytest.mark.parametrize("year, month, filter", ([0, np.arange(0, 3), np.arange(2, 5)],
                                                 [1, np.arange(0, 4), np.arange(1, 5)],
                                                 [2, np.arange(0, 5), np.arange(5)],
                                                 [4, np.arange(2, 7), np.arange(5)],
                                                 [7, np.arange(5, 10), np.arange(5)],
                                                 [8, np.arange(-4, 0), np.arange(0, 4)],
                                                 [9, np.arange(-3, 0), np.arange(0, 3)]))
def test_get_filter_ranges(year: int,
                           month: np.ndarray,
                           filter: np.ndarray) -> None:
    """Test that the array ranges are generated correctly"""

    all_years = np.arange(2000, 2010, 1)

    month_range, filter_range = climatological.get_filter_ranges(year, all_years)

    np.testing.assert_almost_equal(month_range, month)
    np.testing.assert_almost_equal(filter_range, filter)


@pytest.mark.parametrize("year, month, filter", ([0, np.arange(0, 3), np.arange(2, 5)],
                                                 [1, np.arange(0, 3), np.arange(1, 4)],
                                                 [2, np.arange(-3, 0), np.arange(0, 3)]))
def test_get_filter_ranges_short(year: int,
                           month: np.ndarray,
                           filter: np.ndarray) -> None:
    """Test that the array ranges are generated correctly
    if there are shorter than 5 years of data"""

    all_years = np.arange(2000, 2003, 1)  # 3 years of data

    month_range, filter_range = climatological.get_filter_ranges(year, all_years)

    np.testing.assert_almost_equal(month_range, month)
    np.testing.assert_almost_equal(filter_range, filter)


def test_get_filter_ranges_single() -> None:
    """Test that the array ranges are generated correctly
    if only a single year is selected"""
    # expected values
    month = np.array([0]) # first and only entry
    filter = np.array([2]) #  centre of 5 point array

    all_years = np.arange(2000, 2001, 1)

    month_range, filter_range = climatological.get_filter_ranges(0, all_years)

    np.testing.assert_almost_equal(month_range, month)
    np.testing.assert_almost_equal(filter_range, filter)


@patch("climatological.get_filter_ranges")
@patch("climatological.get_weights")
def test_low_pass_filter_zero_sum(get_weights_mock: Mock,
                                  get_filter_mock: Mock) -> None:
    """Test process through low pass filter routine if no data
    to generate weights"""

    station = _setup_station(nyears=1)

    ann_anoms = np.array([0]) # single year, set to zero on purpose
    anoms = np.arange(24*31) # single month

    # dummy return value
    get_filter_mock.return_value = (0, 0)

    result = climatological.low_pass_filter(anoms, station, ann_anoms, 1)

    get_weights_mock.assert_not_called()
    get_filter_mock.assert_called_once_with(0, np.array([2000]))
    np.testing.assert_almost_equal(result, anoms)


@patch("climatological.get_filter_ranges")
@patch("climatological.get_weights")
def test_low_pass_filter(get_weights_mock: Mock,
                         get_filter_mock: Mock) -> None:
    """Test process through low pass filter routine"""

    station = _setup_station(nyears=1)

    ann_anoms = np.array([10]) # single year
    anoms = np.arange(24*31) # single month

    # dummy return value
    get_filter_mock.return_value = (0, 0)
    get_weights_mock.return_value = 10

    result = climatological.low_pass_filter(anoms, station, ann_anoms, 1)

    get_weights_mock.assert_called_once_with(ann_anoms, 0, 0)
    get_filter_mock.assert_called_once_with(0, np.array([2000]))
    np.testing.assert_almost_equal(result,
                                   np.arange(24*31) - get_weights_mock.return_value)


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
                                        np.logical_and(
                                            station.months == 1,
                                            station.hours == 1)
                                        ))

    result = climatological.calculate_climatology(station.temperature,
                                                  hmlocs)

    assert isinstance(result, np.ma.MaskedArray)
    expected = np.ma.array(0, mask=True)

    # as all elements masked, need to test separately
    np.testing.assert_array_almost_equal(result.data, expected.data)
    np.testing.assert_array_almost_equal(result.mask, expected.mask)


def test_calculate_anomalies() -> None:
    """Test if anomalies created correctly"""
    station = _setup_station()

    # this will get edited in place
    result = np.ma.zeros(station.temperature.data.shape[0])
    result.mask = np.ones(result.shape[0])  # all masked

    climatological.calculate_anomalies(station,
                                       station.temperature,
                                       result,
                                       1,
                                       winsorize=False)

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
                                                       station.temperature.data,
                                                       1)
    assert result.shape[0] == 10

    expected = np.ma.ones(10) * qc_utils.average(np.arange(24))

    np.testing.assert_almost_equal(result, expected)


def test_prepare_data_no_data() -> None:
    """Test data preparation steps, with too little data"""
    station = _setup_station(nyears=4)

    result = climatological.prepare_data(station,
                                         station.temperature,
                                         1)

    expected = np.ma.zeros(station.temperature.data.shape[0])
    expected.mask = np.ones(expected.shape[0])

    # insufficient data, all anomalies zero, and masked
    np.testing.assert_almost_equal(result, expected)


@patch("climatological.calculate_anomalies")
def test_prepare_data_no_anoms(calc_anoms_mock: Mock) -> None:
    """Test data preparation steps, with too little data"""
    station = _setup_station()

    # mask all data, so resulting anomalies will all be masked
    station.temperature.data.mask[:] = True

    result = climatological.prepare_data(station,
                                         station.temperature,
                                         1)

    # anomalies will have been modified in place, changing mask
    expected = np.ma.zeros(station.temperature.data.shape[0])
    expected.mask = np.ones(expected.shape[0])
    mlocs, = np.nonzero(station.months == 1)
    expected.mask[mlocs] = False

    np.testing.assert_almost_equal(result, expected)
    calc_anoms_mock.assert_called_once()


@patch("climatological.low_pass_filter")
@patch("climatological.calculate_annual_anomalies")
@patch("climatological.normalise_anomalies")
@patch("climatological.calculate_anomalies")
def test_prepare_data(calc_anoms_mock: Mock,
                      norm_anoms_mock: Mock,
                      ann_anoms_mock: Mock,
                      lpf_mock: Mock) -> None:
    """Test data preparation steps"""
    station = _setup_station()
    mlocs, = np.nonzero(station.months == 1)

    # mask all data, so resulting anomalies will all be masked
    station.temperature.data.mask[:] = True

    norm_anoms_mock.return_value = np.arange(mlocs.shape[0])
    ann_anoms_mock.return_value = np.arange(np.unique(station.years).shape[0])

    result = climatological.prepare_data(station,
                                         station.temperature,
                                         1)

    # anomalies will have been modified in place, changing mask
    expected_anoms = np.ma.zeros(station.temperature.data.shape[0])
    expected_anoms.mask = np.ones(expected_anoms.shape[0])

    expected_anoms.mask[mlocs] = False

    # check normalise_anomalies calls
    norm_anoms_mock.assert_called_once()
    calls = norm_anoms_mock.call_args_list[0]
    np.testing.assert_array_almost_equal(calls.args[0], expected_anoms)
    np.testing.assert_array_almost_equal(calls.args[1], mlocs)

    # check calculate_annual_anomalies calls
    ann_anoms_mock.assert_called_once()
    calls = ann_anoms_mock.call_args_list[0]
    assert calls.args[0] == station
    np.testing.assert_array_almost_equal(calls.args[1], norm_anoms_mock.return_value)
    assert calls.args[2] == 1

    # check low_pass_filter calls
    lpf_mock.assert_called_once()
    calls = lpf_mock.call_args_list[0]
    np.testing.assert_array_almost_equal(calls.args[0], norm_anoms_mock.return_value)
    assert calls.args[1] == station
    np.testing.assert_array_almost_equal(calls.args[2], ann_anoms_mock.return_value)
    assert calls.args[3] == 1


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