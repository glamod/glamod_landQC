"""
Contains tests for variance.py
"""
import numpy as np
import pandas as pd
import datetime as dt
import pytest
from unittest.mock import patch, Mock, call

import variance
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
    variances = np.ma.arange(length)

    prepare_data_mock.return_value = variances

    station = _setup_station()
    config_dict = {}

    _ = variance.find_thresholds(station.temperature,
                                 station, config_dict,
                                 winsorize=False)

    assert config_dict["VARIANCE-temperature"]["1-average"] == qc_utils.average(variances)
    #  all the same in this example
    assert config_dict["VARIANCE-temperature"]["1-spread"] == qc_utils.spread(variances)


@patch("variance.prepare_data")
def test_find_thresholds_short(prepare_data_mock: Mock) -> None:
    """Test writing of config dictionary correct given unable to calculate average and spread
       as data length is too short"""

    length = 9
    variances = np.ma.arange(length)

    prepare_data_mock.return_value = variances

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
    variances = np.ma.arange(length)

    prepare_data_mock.return_value = variances

    station = _setup_station()
    config_dict = {"VARIANCE-temperature": {
        "1-average" : 1,
        "1-spread" : 2
    }
                   }

    result_bad, result_var = variance.identify_bad_years(station.temperature,
                                                         station, config_dict, 1,
                                                         winsorize=False)

    np.testing.assert_array_equal(result_var, (variances - 1)/2)

    expected_bad, = np.nonzero(((variances-1)/2) > 8)
    np.testing.assert_array_equal(result_bad, expected_bad)


@patch("variance.prepare_data")
def test_identify_bad_years_mdi(prepare_data_mock: Mock) -> None:
    """Simple test of routine to identify which years (indices) exceed range"""

    length = 20
    variances = np.ma.arange(length)

    prepare_data_mock.return_value = variances

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


@patch("variance.read_wind_or_pressure")
def test_check_if_storm_no_data(read_mock: Mock) -> None:
    """Test that insufficient data return from reading returns correctly"""

    station = _setup_station(varname = "sea_level_pressure")

    # set up MetVar and put into station
    wind = utils.MeteorologicalVariable("wind_speed",
                                       -99.9, "m/s", "float")
    wind.store_data(np.ma.copy(station.sea_level_pressure.data))
    wind.store_flags(np.copy(station.sea_level_pressure.flags))
    setattr(station, "wind_speed", wind)

    month_locs = np.arange(wind.data.shape[0])
    year_locs = np.arange(24*31)

    read_mock.return_value = (-1, -1)

    result = variance.check_if_storm(station, wind, month_locs, year_locs)

    # check calls use correct data
    for each_call, expected_call in zip(read_mock.call_args_list,
                                        [wind.data[month_locs],
                                         station.sea_level_pressure.data[month_locs]]):

        np.testing.assert_array_equal(each_call.args[0], expected_call)
    # and the final result
    assert result == False


@patch("variance.read_wind_or_pressure")
def test_check_if_storm_no_year_data(read_mock: Mock) -> None:
    """Test that insufficient data return from reading returns correctly"""

    station = _setup_station(varname = "sea_level_pressure")

    # set up MetVar and put into station
    station.sea_level_pressure.data.mask[100:] = True

    wind = utils.MeteorologicalVariable("wind_speed",
                                       -99.9, "m/s", "float")
    wind.store_data(station.sea_level_pressure.data)
    wind.store_flags(np.copy(station.sea_level_pressure.flags))
    setattr(station, "wind_speed", wind)

    month_locs = np.arange(wind.data.shape[0])
    year_locs = np.arange(24*31)

    # set so averages have no impact on scaling for this test
    read_mock.return_value = (0, 1)

    result = variance.check_if_storm(station, wind, month_locs, year_locs)

    assert result == False


@patch("variance.sequential_differences")
@patch("variance.high_wind_low_pressure_match")
@patch("variance.read_wind_or_pressure")
def test_check_if_storm_no_couldbe(read_mock: Mock,
                                   match_mock: Mock,
                                   seqdif_mock: Mock) -> None:
    """Test that if no couldbe_storm, then returns correctly"""

    station = _setup_station(varname="sea_level_pressure")

    # set up MetVar and put into station
    wind = utils.MeteorologicalVariable("wind_speed",
                                       -99.9, "m/s", "float")
    wind.store_data(np.ma.copy(station.sea_level_pressure.data))
    wind.store_flags(np.copy(station.sea_level_pressure.flags))
    setattr(station, "wind_speed", wind)

    month_locs = np.arange(wind.data.shape[0])
    year_locs = np.arange(24*31)

    # set so averages have no impact on scaling for this test
    read_mock.return_value = (0, 1)
    seqdif_mock.return_value = False

    result = variance.check_if_storm(station, wind, month_locs, year_locs)

    # check calls use correct data
    calls = match_mock.call_args_list[0]
    # only the single year of data
    np.testing.assert_array_equal(calls.args[0], wind.data[year_locs])
    # and the inverse for the pressure data scaling, hence the "-"
    np.testing.assert_array_equal(calls.args[1],
                                  -station.sea_level_pressure.data[year_locs])

    # check result as expected (flags retained)
    np.testing.assert_array_equal(result, year_locs)


@patch("variance.sequential_differences")
@patch("variance.read_wind_or_pressure")
def test_check_if_storm_couldbe(read_mock: Mock,
                                   seqdif_mock: Mock) -> None:
    """Test that if couldbe_storm, then returns correctly"""

    station = _setup_station(varname="sea_level_pressure")

    # set up MetVar and put into station
    wind = utils.MeteorologicalVariable("wind_speed",
                                       -99.9, "m/s", "float")
    wind.store_data(np.ma.copy(station.sea_level_pressure.data))
    wind.store_flags(np.copy(station.sea_level_pressure.flags))
    setattr(station, "wind_speed", wind)

    month_locs = np.arange(wind.data.shape[0])
    year_locs = np.arange(24*31)

    # set so averages have no impact on scaling for this test
    read_mock.return_value = (0, 1)
    seqdif_mock.return_value = True

    result = variance.check_if_storm(station, wind, month_locs, year_locs)

    # check that result as it should be (flags cleared)
    np.testing.assert_array_equal(result, np.array([]))


@patch("variance.logger")
@patch("variance.identify_bad_years")
def test_variance_check_none(identify_mock: Mock,
                             logger_mock: Mock) -> None:
    """Test flow through main routine when no data"""
    station = _setup_station()
    config_dict = {}

    identify_mock.return_value = np.array([]), np.array([])

    variance.variance_check(station.temperature, station,
                            config_dict)

    calls = [call("Variance temperature"),
             call("   Cumulative number of flags set: 0")]
    logger_mock.info.assert_has_calls(calls)


@patch("variance.logger")
@patch("variance.identify_bad_years")
def test_variance_check(identify_mock: Mock,
                        logger_mock: Mock) -> None:
    """Test flow through main control routine"""
    station = _setup_station()
    config_dict = {}

    identify_mock.return_value = np.array([0]), np.arange(10)

    variance.variance_check(station.temperature, station,
                            config_dict)

    calls = [call("Variance temperature"),
             call(f"   Cumulative number of flags set: {24*31}")]
    logger_mock.info.assert_has_calls(calls)

    assert np.all(station.temperature.flags[:744] == "V")
    assert np.all(station.temperature.flags[744:] == "")


@patch("variance.logger")
@patch("variance.identify_bad_years")
@patch("variance.check_if_storm")
def test_variance_check_no_storm(storm_mock: Mock,
                                identify_mock: Mock,
                                logger_mock: Mock) -> None:
    """Test that flow handled if storm checking called, but no storm found"""
    station = _setup_station(varname = "sea_level_pressure")
    config_dict = {}

    identify_mock.return_value = np.array([0]), np.arange(10)
    storm_mock.return_value = False
    variance.variance_check(station.sea_level_pressure, station,
                            config_dict)

    calls = [call("Variance sea_level_pressure"),
             call(f"   Cumulative number of flags set: {24*31}")]
    logger_mock.info.assert_has_calls(calls)

    # As storm is False, first month is a bad year, flags should be set
    assert np.all(station.sea_level_pressure.flags[:744] == "V")
    assert np.all(station.sea_level_pressure.flags[744:] == "")


@patch("variance.logger")
@patch("variance.identify_bad_years")
@patch("variance.check_if_storm")
def test_variance_check_storm_found(storm_mock: Mock,
                                    identify_mock: Mock,
                                    logger_mock: Mock) -> None:
    """Test that flow handled if storm checking called, but no storm found"""
    station = _setup_station(varname = "sea_level_pressure")
    config_dict = {}

    identify_mock.return_value = np.array([0]), np.arange(10)
    storm_mock.return_value = np.array([])
    variance.variance_check(station.sea_level_pressure, station,
                            config_dict)

    calls = [call("Variance sea_level_pressure"),
             call("   Cumulative number of flags set: 0")]
    logger_mock.info.assert_has_calls(calls)

    # As storm is True (returns array), first month is a bad year, flags should NOT be set
    assert np.all(station.sea_level_pressure.flags[:744] == "")
    assert np.all(station.sea_level_pressure.flags[744:] == "")


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