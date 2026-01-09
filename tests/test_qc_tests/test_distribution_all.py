"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock

import distribution_all

import common
import utils
import qc_utils

def _setup_station(indata: np.ma.MaskedArray) -> utils.Station:
    """Create a station object to hold the information enabling
    the QC test to be tested

    Parameters
    ----------
    indata : np.ma.MaskedArray
        dummy temperature data

    Returns
    -------
    utils.Station
        Station with appropriate attributes
    """
    # set up the data
    indata.mask = np.zeros(len(indata))

    # make MetVars
    temperature = common.example_test_variable("temperature", indata)

    # make Station
    station = common.example_test_station(temperature)

    return station


def test_find_monthly_scaling() -> None:
    """Test calculation and storage of scaling parameters"""

    indata = np.ma.array(np.arange(200))
    config_dict = {}

    result = distribution_all.find_monthly_scaling(indata, config_dict,
                                                   "temperature", 1)

    assert result[0] == qc_utils.average(indata)
    assert result[1] == qc_utils.spread(indata)

    assert config_dict["ADISTRIBUTION-temperature"]["1-clim"] == result[0]
    assert config_dict["ADISTRIBUTION-temperature"]["1-spread"] == result[1]


def test_find_monthly_scaling_no_data() -> None:
    """Test calculation and storage of scaling parameters"""

    indata = np.ma.array(np.arange(10))
    config_dict = {}

    result = distribution_all.find_monthly_scaling(indata, config_dict,
                                                   "temperature", 1)

    assert result[0] == utils.MDI
    assert result[1] == utils.MDI

    assert config_dict["ADISTRIBUTION-temperature"]["1-clim"] == result[0]
    assert config_dict["ADISTRIBUTION-temperature"]["1-spread"] == result[1]


@patch("distribution_all.find_monthly_scaling")
def test_prepare_all_data_full_calls_mdi(find_monthly_mock: Mock) -> None:
    """Test when doing a full run, and with MDI as returns"""

    find_monthly_mock.return_value = (utils.MDI, utils.MDI)

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               {}, full=True)

    # test calls
    find_monthly_mock.assert_called_once()
    calls = find_monthly_mock.call_args_list[0]
    np.testing.assert_array_equal(calls.args[0],
                                  station.temperature.data)
    assert calls.args[1] == {}
    assert calls.args[2] == "temperature"
    assert calls.args[3] == 1

    # test result
    np.testing.assert_almost_equal(result, np.ma.array([utils.MDI]))


def test_prepare_all_data_spread_zero() -> None:
    """Test when reading from dict, with spread of zero"""

    temperature_values = [("1-clim", 1)]
    temperature_values += [("1-spread", 0)]
    config_dict = {"ADISTRIBUTION-temperature": dict(temperature_values)}

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               config_dict, full=False)

    # test result
    np.testing.assert_almost_equal(result,
                                   np.ma.arange(10)-1)


def test_prepare_all_data_spread_nonzero() -> None:
    """Test when reading from dict, with nonzero spread"""

    temperature_values = [("1-clim", 1)]
    temperature_values += [("1-spread", 2)]
    config_dict = {"ADISTRIBUTION-temperature": dict(temperature_values)}

    station = _setup_station(np.ma.arange(10))
    result = distribution_all.prepare_all_data(station.temperature,
                                               station, 1,
                                               config_dict, full=False)

    # test result
    np.testing.assert_almost_equal(result,
                                   (np.ma.arange(10)-1)/2)


def test_write_thresh_to_config_dict() -> None:
    """Test that writing of dictionary works"""
    config_dict = {}

    distribution_all.write_thresh_to_config_dict(config_dict, "temperature",
                                                 1, 10., -10.)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == 10
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == -10


def test_write_thresh_to_config_dict_with_exist() -> None:
    """Test that writing of dictionary works"""
    config_dict = {"ADISTRIBUTION-temperature" : {}}

    distribution_all.write_thresh_to_config_dict(config_dict, "temperature",
                                                 1, 10., -10.)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == 10
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == -10


@patch("distribution_all.prepare_all_data")
def test_find_thresholds_mdi(prepare_mock: Mock) -> None:
    """Test stored values correct if no scaling possible"""
    station = _setup_station(np.ma.arange(10))
    prepare_mock.return_value = np.ma.array([utils.MDI])
    config_dict = {}

    distribution_all.find_thresholds(station.temperature,
                                     station, config_dict)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == utils.MDI
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == utils.MDI


@patch("distribution_all.prepare_all_data")
def test_find_thresholds_single_value(prepare_mock: Mock) -> None:
    """Test stored values correct if no fitting possible"""

    station = _setup_station(np.ma.arange(10))
    prepare_mock.return_value = np.ma.array([0, 0, 0, 0, 0, 0])
    config_dict = {}

    distribution_all.find_thresholds(station.temperature,
                                     station, config_dict)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == utils.MDI
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == utils.MDI


@patch("distribution_all.prepare_all_data")
@patch("distribution_all.qc_utils.create_bins")
@patch("distribution_all.np.histogram")
@patch("distribution_all.qc_utils.fit_gaussian")
@patch("distribution_all.qc_utils.skew_gaussian")
def test_find_month_thresholds_fitting_no_thresh(gauss_mock: Mock,
                                                 fit_mock: Mock,
                                                 hist_mock: Mock,
                                                 create_bins_mock: Mock,
                                                 prepare_mock: Mock) ->  None:
    """Test thresholds set at extremal values
    if curve doesn't fall below threshold"""

    station = _setup_station(np.ma.arange(10))
    config_dict = {}

    # want data only for first month, so make something to generate that
    prepare_mock.side_effect = [station.temperature.data if i==0 else np.ma.MaskedArray([10]) for i in range(12)]

    # simple bins and histogram
    bins = np.arange(-6., 7., 1)
    create_bins_mock.return_value = bins

    # only need this to fill unused variables
    hist_mock.return_value = (np.arange(10), None)
    # mocking this so that a skew Gaussian called correctly
    fit_mock.return_value = np.array([1, 0, 1, 0.1])

    # mocking values for the fitted Gaussian
    #    Bins < 0.1 flagged, but rounded down/up to be inclusive
    #    so before (lower) or after (higher) the 0.05 values
    gauss_mock.return_value = np.array([0.15, 0.2, 0.5, 0.5, 1, 3, 5, 3, 1, 0.5, 0.2, 0.15])

    distribution_all.find_thresholds(station.temperature,
                                     station, config_dict)

    calls = gauss_mock.call_args_list[0]
    np.testing.assert_almost_equal(calls.args[0], bins[1:]-0.5)
    np.testing.assert_almost_equal(calls.args[1],fit_mock.return_value)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == 6.
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == -6.


@patch("distribution_all.prepare_all_data")
@patch("distribution_all.qc_utils.create_bins")
@patch("distribution_all.np.histogram")
@patch("distribution_all.qc_utils.fit_gaussian")
@patch("distribution_all.qc_utils.skew_gaussian")
def test_find_month_thresholds_fitting(gauss_mock: Mock,
                                       fit_mock: Mock,
                                       hist_mock: Mock,
                                       create_bins_mock: Mock,
                                       prepare_mock: Mock) ->  None:
    """Test how the thresholds are determined when selecting the
    fitted curve falls below the set level"""

    station = _setup_station(np.ma.arange(10))
    config_dict = {}

    # want data only for first month, so make something to generate that
    prepare_mock.side_effect = [station.temperature.data if i==0 else np.ma.MaskedArray([10]) for i in range(12)]

    # simple bins and histogram
    bins = np.arange(-6., 7., 1)
    create_bins_mock.return_value = bins

    # only need this to fill unused variables
    hist_mock.return_value = (np.arange(10), None)
    # mocking this so that a skew Gaussian called correctly
    fit_mock.return_value = np.array([1, 0, 1, 0.1])

    # mocking values for the fitted Gaussian
    #    Bins < 0.1 flagged, but rounded down/up to be inclusive
    #    so before (lower) or after (higher) the 0.05 values
    gauss_mock.return_value = np.array([0.01, 0.05, 0.5, 0.5, 1, 3, 5, 3, 1, 0.5, 0.05, 0.01])

    distribution_all.find_thresholds(station.temperature,
                                     station, config_dict)

    calls = gauss_mock.call_args_list[0]
    np.testing.assert_almost_equal(calls.args[0], bins[1:]-0.5)
    np.testing.assert_almost_equal(calls.args[1],fit_mock.return_value)

    assert config_dict["ADISTRIBUTION-temperature"]["1-uthresh"] == 4.5
    assert config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] == -4.5


@patch("distribution_all.dist_monthly.prepare_monthly_data")
def test_average_and_spread(prepare_mock: Mock) -> None:
    """Test that values from mocked monthly averages are returned as expected"""
    # dummy station
    station = _setup_station(np.ma.arange(10))

    prepare_mock.return_value = np.ma.arange(5)

    av, sp = distribution_all.average_and_spread(station.temperature,
                                                 station, 1)

    assert av == qc_utils.average(np.ma.arange(5))
    assert sp == qc_utils.spread(np.ma.arange(5))



@patch("distribution_all.average_and_spread")
def test_find_storms_few_data(av_and_sp_mock: Mock) -> None:
    """Test routine exits when there's insufficient data"""
    station = _setup_station(np.ma.arange(10))
    wind_speed = common.example_test_variable("wind_speed",
                                              np.ma.arange(10))
    station_pressure = common.example_test_variable("station_level_pressure",
                                                    np.ma.arange(10))

    station.station_level_pressure = station_pressure
    station.wind_speed = wind_speed

    distribution_all.find_storms(station, station_pressure,
                                    1, np.array([]))

    # data are too short so routine should have exited
    av_and_sp_mock.assert_not_called()


@patch("distribution_all.average_and_spread")
@patch("distribution_all.check_through_storms")
def test_find_storms(storm_check_mock: Mock,
                     av_and_sp_mock: Mock) -> None:
    """Test routine finds single storm from wind and pressure values"""

    # set up the station and data for a single January
    indata = np.ma.ones(31*24)*2
    station = _setup_station(indata)

    # windier than normal on days 2 and 26
    indata[24: 48] = 20
    indata[25*24: 26*24] = 20
    wind_speed = common.example_test_variable("wind_speed",
                                              indata)
    station.wind_speed = wind_speed

    indata = np.ma.ones(31*24)*100
    # lower pressure than normal on days 2 and 13
    indata[24: 48] = 10
    indata[12*24: 13*24] = 10
    station_pressure = common.example_test_variable("station_level_pressure",
                                                    indata)
    station.station_level_pressure = station_pressure

    # set initial flags for pressure, days 2 and 13
    flags = np.array(["" for i in range(31*24)])
    flags[24: 48] = "d"
    flags[12*24: 13*24] = "d"

    # mock return values for the average and spread; wind, then pressure
    av_and_sp_mock.side_effect = [(2, 2), (80, 10)]

    expected_storms = np.arange(24, 48)  # day 2 only
    storm_check_mock.return_value = expected_storms

    distribution_all.find_storms(station, station_pressure,
                                    1, flags)

    expected_flags = np.array(["" for i in range(31*24)])
    expected_flags[12*24: 13*24] = "d"  # day 13 only

    # test that routine called with appropriate info
    calls = storm_check_mock.call_args_list[0]
    np.testing.assert_array_equal(calls.args[0], expected_storms)
    np.testing.assert_almost_equal(calls.args[1], wind_speed.data)

    np.testing.assert_array_equal(flags, expected_flags)



@patch("distribution_all.prepare_all_data")
@patch("distribution_all.qc_utils.create_bins")
@patch("distribution_all.np.histogram")
@patch("distribution_all.qc_utils.find_gap")
def test_all_obs_gap(find_mock: Mock,
                     hist_mock: Mock,
                     create_bins_mock: Mock,
                     prepare_mock: Mock) ->  None:
    """Test how the thresholds are used to set flags"""

    # set up station
    station = _setup_station(np.ma.ones(31*24))

    # set up config dictionary
    config_dict = {"ADISTRIBUTION-temperature": {"1-uthresh": 5.}}
    config_dict["ADISTRIBUTION-temperature"]["1-lthresh"] = -5.

    # generate some dummy anomalies, with values that can be flagged
    anomalies = np.ma.array(np.random.normal(0.0, 1.0, station.temperature.data.shape[0]))
    anomalies[:10] = 100
    anomalies[-10:] = -100
    prepare_mock.side_effect = [anomalies if i==0 else np.ma.MaskedArray([utils.MDI]) for i in range(12)]

    # simple bins and histogram, so that routine flows
    bins = np.arange(-6., 7, 1)
    create_bins_mock.return_value = bins
    # only need this to fill unused variables
    hist_mock.return_value = (np.arange(10),
                              None)

    # gap check tested elsewhere, so return values which will act to flag
    find_mock.side_effect = [10, -10]

    distribution_all.all_obs_gap(station.temperature, station, config_dict)

    # build the expected array
    expected_flags = np.array(["" for _ in station.times])
    expected_flags[:10] = "d"
    expected_flags[-10:] = "d"

    np.testing.assert_equal(station.temperature.flags, expected_flags)


@patch("distribution_all.prepare_all_data")
@patch("distribution_all.qc_utils.create_bins")
@patch("distribution_all.np.histogram")
@patch("distribution_all.qc_utils.find_gap")
@patch("distribution_all.find_storms")
def test_all_obs_gap_pressure(storms_mock: Mock,
                              find_mock: Mock,
                              hist_mock: Mock,
                              create_bins_mock: Mock,
                              prepare_mock: Mock) ->  None:
    """Test how the thresholds are used to set flags"""

    # make Station
    slp = common.example_test_variable("sea_level_pressure", np.ma.ones(31*24))
    station = common.example_test_station(slp)

    # set up config dictionary
    config_dict = {"ADISTRIBUTION-sea_level_pressure": {"1-uthresh": 5.}}
    config_dict["ADISTRIBUTION-sea_level_pressure"]["1-lthresh"] = -5.

    # generate some dummy anomalies, with values that can be flagged
    anomalies = np.ma.array(np.random.normal(0.0, 1.0, station.sea_level_pressure.data.shape[0]))
    anomalies[:10] = 100
    anomalies[-10:] = -100
    prepare_mock.side_effect = [anomalies if i==0 else np.ma.MaskedArray([utils.MDI]) for i in range(12)]

    # simple bins and histogram, so that routine flows
    bins = np.arange(-6., 7, 1)
    create_bins_mock.return_value = bins
    # only need this to fill unused variables
    hist_mock.return_value = (np.arange(10),
                              None)

    # gap check tested elsewhere, so return values which will act to flag
    find_mock.side_effect = [10, -10]

    distribution_all.all_obs_gap(station.sea_level_pressure, station, config_dict)

    # finally, test this was called
    expected_flags = np.array(["" for _ in station.times])
    expected_flags[:10] = "d"
    expected_flags[-10:] = "d"

    # Check that storms routine called as expected
    calls = storms_mock.call_args_list[0]
    assert calls.args[0] == station
    assert calls.args[1] == station.sea_level_pressure
    assert calls.args[2] == 1
    np.testing.assert_array_equal(calls.args[3], expected_flags)


@patch("distribution_all.find_thresholds")
@patch("distribution_all.all_obs_gap")
def test_dgc(all_gap_mock: Mock,
             thresholds_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    distribution_all.dgc(station, ["temperature"], {},
                         full=True, plots=True)

    # Mock to check call occurs as expected with right return
    thresholds_mock.assert_called_once_with(station.temperature,
                                            station, {},
                                            full=True,
                                            plots=True,
                                            diagnostics=False)

    all_gap_mock.assert_called_once_with(station.temperature,
                                         station, {},
                                         plots=False,
                                         diagnostics=False)