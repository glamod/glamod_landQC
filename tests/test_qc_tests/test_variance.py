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


def test_calculate_climatology_nodata() -> None:
    """Test climatology calculation if too short data length"""
    indata = np.ma.ones(20)

    clim, c_mask = variance.calculate_climatology(indata,
                                                  winsorize=False)

    assert c_mask  # test if False
    assert clim == 0


def test_calculate_hourly_anomalies() -> None:

    # station has values of 0..23 repeating for each hour
    station = _setup_station()

    result = variance.calculate_hourly_anomalies(station.hours,
                                                 station.temperature.data)

    # hence anomalies will all be zero, as each hour of a day
    #    the same across the month
    assert np.all(result == 0)


def test_calculate_hourly_anomalies_nonzero() -> None:

    # station has values of 0..23 repeating for each hour
    #   10 Januaries worth
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
    """Test normalising step when using calculated spread"""

    anomalies = np.ma.arange(20)

    result = variance.normalise_hourly_anomalies(anomalies)

    spread = qc_utils.spread(anomalies)

    np.testing.assert_allclose(result, anomalies/spread)


def test_normalise_hourly_anomalies_small_spread() -> None:
    """Test normalising when spread too small, and hence set to 1.5"""

    anomalies = np.ma.ones(20)

    result = variance.normalise_hourly_anomalies(anomalies)

    np.testing.assert_allclose(result, anomalies/1.5)


def test_normalise_hourly_anomalies_short() -> None:
    """Test normalising when too short data, and hence set to 1.5"""

    anomalies = np.ma.ones(5)

    result = variance.normalise_hourly_anomalies(anomalies)

    np.testing.assert_allclose(result, anomalies/1.5)


def test_calculate_yearly_variances() -> None:
    """Test the calculation of variances for a given month over all years"""

    # station has values of 0..23 repeating for each hour
    # so variance is just that of [00..23]
    station = _setup_station()
    locs = np.arange(station.years.shape[0])

    result = variance.calculate_yearly_variances(station.years,
                                                 station.temperature.data,
                                                 locs)

    expected=qc_utils.spread(np.arange(24))
    assert np.all(result == expected)
    assert len(result) == 10


@patch("variance.calculate_hourly_anomalies")
@patch("variance.normalise_hourly_anomalies")
@patch("variance.calculate_yearly_variances")
def test_prepare_data(yearly_var_mock: Mock,
                      norm_anoms_mock: Mock,
                      hourly_anoms_mock: Mock) -> None:
    """Test call of processing scripts done correctly"""

    station = _setup_station()
    temperature = station.temperature

    _ = variance.prepare_data(temperature, station, 1)

    hourly_anoms_mock.assert_called_once()
    norm_anoms_mock.assert_called_once()
    yearly_var_mock.assert_called_once()


@patch("variance.prepare_data")
def test_find_thresholds(prepare_data_mock: Mock) -> None:
    """Test writing of config dictionary correct given mocked variances"""
    length = 20
    vars = np.ma.arange(length)

    prepare_data_mock.return_value = vars

    station = _setup_station()
    config_dict = {}

    _ = variance.find_thresholds(station.temperature,
                                 station, config_dict,
                                 winsorize=False)

    assert config_dict["VARIANCE-temperature"]["1-average"] == qc_utils.average(vars)
    #  all the same in this example
    assert config_dict["VARIANCE-temperature"]["1-spread"] == qc_utils.spread(vars)


@patch("variance.prepare_data")
def test_find_thresholds_short(prepare_data_mock: Mock) -> None:
    """Test writing of config dictionary correct given unable to calculate average and spread
       as data length is too short"""

    length = 9
    vars = np.ma.arange(length)

    prepare_data_mock.return_value = vars

    station = _setup_station()
    config_dict = {}

    _ = variance.find_thresholds(station.temperature,
                                 station, config_dict,
                                 winsorize=False)

    assert config_dict["VARIANCE-temperature"]["1-average"] == utils.MDI
    #  all the same in this example
    assert config_dict["VARIANCE-temperature"]["1-spread"] == utils.MDI


@patch("variance.prepare_data")
def test_identify_bad_years(prepare_data_mock: Mock) -> None:
    """Simple test of routine to identify which years (indices) exceed range"""

    length = 20
    vars = np.ma.arange(length)

    prepare_data_mock.return_value = vars

    station = _setup_station()
    config_dict = {"VARIANCE-temperature": {
        "1-average" : 1,
        "1-spread" : 2
    }
                   }

    result_bad, result_var = variance.identify_bad_years(station.temperature,
                                                         station, config_dict, 1,
                                                         winsorize=False)

    np.testing.assert_array_equal(result_var, (vars - 1)/2)

    expected_bad, = np.nonzero(((vars-1)/2) > 8)
    np.testing.assert_array_equal(result_bad, expected_bad)


@patch("variance.prepare_data")
def test_identify_bad_years_mdi(prepare_data_mock: Mock) -> None:
    """Simple test of routine to identify which years (indices) exceed range"""

    length = 20
    vars = np.ma.arange(length)

    prepare_data_mock.return_value = vars

    station = _setup_station()
    config_dict = {"VARIANCE-temperature": {
        "1-average" : utils.MDI,
        "1-spread" : utils.MDI
    }
                   }

    result_bad, result_var = variance.identify_bad_years(station.temperature,
                                                         station, config_dict, 1,
                                                         winsorize=False)

    np.testing.assert_array_equal(result_var, np.array([]))
    np.testing.assert_array_equal(result_bad, np.array([]))


def test_read_wind_or_pressure_short() -> None:
    """Test passing of array returns correct values for spread and average"""

    indata = np.ma.arange(20)

    result_avg, result_spread = variance.read_wind_or_pressure(indata)

    assert result_avg == -1
    assert result_spread == -1


@patch("utils.DATA_COUNT_THRESHOLD", 10)
def test_read_wind_or_pressure() -> None:
    """Test passing of array returns correct values for spread and average"""
    # override the minimum data count
    indata = np.ma.ones(10)

    result_avg, result_spread = variance.read_wind_or_pressure(indata)

    assert result_avg == 1
    assert result_spread == 0


def test_high_wind_low_pressure_match() -> None:
    """Test selection of overlapping high winds/low pressures"""
    # set up dummy arrays
    scaled_wind = np.ma.ones(10)
    scaled_pressure = np.ma.ones(10)

    # locations to trigger test
    scaled_wind[:2] = 5
    scaled_pressure[1:3 ] = 5

    result = variance.high_wind_low_pressure_match(scaled_wind, scaled_pressure)

    assert result


def test_high_wind_low_pressure_no_match() -> None:
    """Test selection of no overlap of high winds/low pressures"""
    scaled_wind = np.ma.ones(10)
    scaled_pressure = np.ma.ones(10)

    # non overlapping locations so no triggering
    scaled_wind[:2] = 5
    scaled_pressure[-2:] = 5

    result = variance.high_wind_low_pressure_match(scaled_wind, scaled_pressure)

    assert not result


@pytest.mark.parametrize("length, storm, expected", [(9, False, False),
                                                     (9, True, True),
                                                     (10, False, True),
                                                     (10, True, True)])
def test_sequential_differences(length: int,
                                storm: bool,
                                expected: bool) -> None:
    """Test that counting of sequential differences works as intended"""

    indiffs = np.append(np.ones(length), -np.ones(length))

    result = variance.sequential_differences(indiffs, storm)

    assert result == expected


@pytest.mark.parametrize("length, storm, expected", [(9, False, False),
                                                     (9, True, True),
                                                     (10, False, True),
                                                     (10, True, True)])
def test_sequential_differences_invert(length: int,
                                       storm: bool,
                                       expected: bool) -> None:
    """Test that counting of sequential differences works as intended"""

    indiffs = np.append(-np.ones(length), np.ones(length))

    result = variance.sequential_differences(indiffs, storm)

    assert result == expected


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