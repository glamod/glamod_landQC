"""
Contains tests for diurnal.py
"""
import numpy as np
import datetime as dt
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import diurnal

import common
import utils


def _setup_station(indata: np.ma.MaskedArray,
                   intimes: pd.Series | None = None) -> utils.Station:

    # set up the data
    indata.mask = np.zeros(len(indata))

    # make MetVars
    temperature = common.example_test_variable("temperature", indata)

    # make Station
    station = common.example_test_station(temperature, intimes)

    return station


def test_make_sines() -> None:
    """Test that stacked sine curves created correctly"""
    sines = diurnal.make_sines()

    expected = np.array([0.5, 0.37059048, 0.25, 0.14644661, 0.0669873, 0.01703709,
                         0., 0.01703709, 0.0669873, 0.14644661, 0.25, 0.37059048,
                         0.5, 0.62940952, 0.75, 0.85355339, 0.9330127, 0.98296291,
                         1., 0.98296291, 0.9330127, 0.85355339, 0.75, 0.62940952])

    # test that sine values for first point in each of the 24
    # rows is as expected
    np.testing.assert_array_almost_equal(sines[:, 0], expected)


@pytest.mark.parametrize("minutes, expected", [(np.array([0, 360, 720, 1080]), True),
                                               (np.array([720, 1080]), False),
                                               (np.array([360, 361, 720]), False),])
def test_quartile_check(minutes: np.ndarray,
                        expected: bool) -> None:
    """Test that quartile check returns True if obs in each 3hr quartile"""

    result = diurnal.quartile_check(minutes)

    assert result == expected


def test_make_scaled_sine() -> None:
    """Test that scaling of sine curve occurs as expected"""
    input_sine = diurnal.make_sines()

    maximum = 10
    minimum = -10

    expected = (input_sine * 20) + minimum

    result = diurnal.make_scaled_sine(maximum, minimum)

    assert np.max(result) == maximum
    assert np.min(result) == minimum

    np.testing.assert_array_equal(result, expected)


def test_find_differences() -> None:
    """Test phase differences between sine curves found sensibly"""

    # example data is in total phase with constructed sines
    day_data = np.array([0, 1, 0, -1])
    minutes = np.array([0, 360, 720, 1080])

    result = diurnal.find_differences(day_data, minutes)

    # test first entry is minimum and close to zero
    assert np.min(result) == result[0]
    np.testing.assert_almost_equal(np.min(result), 0)


def test_find_uncertainties() -> None:
    """Test uncertainties are found as expected"""

    differences = np.array([7,7,7,7,7,6,
                            5,4,3,2,1,0,
                            1,1.5,2,3,4,5,
                            6,7,7,7,7,7])

    # hence values under threshold (<3) should be [2,1,0,1,1.5,2]
    #  indices [ 9 10 11 12 13 14]
    # best fit == 11
    # uncertainty dominated by RHS, 14-11+1 results in 4.

    best_fit = np.argmin(differences)

    result = diurnal.find_uncertainty(differences, best_fit)

    assert result == 4


def test_find_fit_and_uncertainty() -> None:
    """Testing calls to calculation routines work as expected"""

    # example data is in total phase with constructed sines
    day_data = np.array([0, 1, 0, -1])
    minutes = np.array([0, 360, 720, 1080])

    fit, _ = diurnal.find_fit_and_uncertainty(day_data, minutes)

    # in total phase with constructed sine
    assert fit == 0
    # uncertainty tested elsewhere


@patch("diurnal.find_fit_and_uncertainty")
def test_get_daily_offset(find_fit_mock: Mock) -> None:
    """Test calculation of fit/uncertainty from station object"""

    find_fit_mock.return_value = (0, 0)

    # set up some random data
    dummy_station = _setup_station(np.ma.arange(24))
    obs_var = dummy_station.temperature

    locs = np.arange(24)

    # no need to test returns, these done elsewhere
    _, _ = diurnal.get_daily_offset(dummy_station, locs, obs_var)

    find_fit_mock.assert_called_once()

    expected_data = np.ma.arange(24)  # data for first 24hr
    expected_data.mask = np.zeros(len(expected_data))
    expected_mins = np.arange(0, 1440, 60)  # minutes

    # testing calls
    np.testing.assert_array_equal(find_fit_mock.call_args[0][0], expected_data)
    np.testing.assert_array_equal(find_fit_mock.call_args[0][1], expected_mins)


def test_get_start_end_ndays() -> None:
    """Test return of start, end and ndays"""

    ndata = 72
    indata = np.ma.arange(ndata)
    start_dt = dt.datetime(2000, 3, 5, 0, 0)
    intimes = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                               for i in range(ndata)])[0])

    dummy_station = _setup_station(indata=indata, intimes=intimes)

    start, end, ndays = diurnal.get_start_end_ndays(dummy_station)

    # despite data starting through a year, the day counter set up
    #   uses complete years
    assert start == dt.date(2000, 1, 1)
    assert end == dt.date(2001, 1, 1)
    assert ndays == 366  # is leap


def test_get_all_daily_offsets() -> None:
    """Test finding diurnal fit and uncertainty for all days in series"""

    # note, uses all possible days, not just days with data

    # set up test data - 3 days worth
    indata = np.ma.zeros(72)
    indata[:] = utils.MDI

    # first day - full sine
    # second day - no data
    # third day - just 4 points
    indata[:24] = (np.sin(2. * np.pi * (np.arange(24)/24.)) + 1.)/2.
    gappy = np.array([48, 54, 60, 66])
    indata[gappy] = [0, 1, 0, -1]

    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    intimes = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                               for i in range(72)])[0])

    # filter
    intimes = intimes[indata != utils.MDI].reset_index(drop=True)
    indata = 10*indata[indata != utils.MDI]

    dummy_station = _setup_station(indata=indata, intimes=intimes)
    obs_var = dummy_station.temperature

    fit, unc = diurnal.get_all_daily_offsets(dummy_station, obs_var)

    assert len(fit) == len(unc) == 366  # 200 is a leap year.
    # second day has no (i.e. insufficient) obs
    np.testing.assert_array_equal(fit[:3], np.array([0, -99, 0]))
    # not testing uncertainties - dealt with elsewhere.


def test_find_best_fit() -> None:
    """Test routine which finds best fit for each uncertainty"""

    uncs = np.arange(1, diurnal.MAX_UNCERTAINTY + 1)
    uncs = np.tile(uncs, 120)

    fits = np.array([11, 12, 13, 14, 15, 16,
                     21, 22, 23, 24, 25, 26,
                     31, 32, 33, 34, 35, 36])
    fits = np.tile(fits, 40)

    result = diurnal.find_best_fit(fits, uncs)

    # should get the median
    expected = np.array([21, 22, 23, 24, 25, 26])

    np.testing.assert_array_equal(result, expected)


@patch("diurnal.get_all_daily_offsets")
@patch("diurnal.find_best_fit")
def test_find_offset(find_best_fit_mock: Mock,
                     daily_offsets_mock: Mock) -> None:
    """Test finding of overall cycle centre"""

    # daily best fits and uncertainties
    daily_offsets_mock.return_value = (np.array([9, 10, 11, 10, 10, 10, 10]),
                                       np.array([2, 2, 2, 3, 4, 5, 6]))

    # best fit for each *uncertainty* value
    find_best_fit_mock.return_value = np.array([-99, 10, 10, 10 ,10, 10])


    # set up other inputs
    indata = np.ma.arange(10)
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    intimes = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                               for i in range(10)])[0])

    dummy_station = _setup_station(indata=indata, intimes=intimes)
    obs_var = dummy_station.temperature
    config_dict = {}
    diurnal.find_offset(obs_var, dummy_station, config_dict)

    assert config_dict[f"DIURNAL-{obs_var.name}"]["peak"] == 10


def test_get_potentially_spurious() -> None:
    """Test selection of days where fit and uncertainty does not include overall fit"""
    best_fit_diurnal = np.array([10, 10, 6, 10, 13, 10, 10])
    best_fit_uncertainty = np.array([2, 2, 2, 3, 3, 3, 3])
    diurnal_offset = 9
    # should highlight where fit +/- uncertainty does not include offset

    result = diurnal.get_potentially_spurious_days(best_fit_diurnal,
                                                   best_fit_uncertainty,
                                                   diurnal_offset)

    np.testing.assert_array_equal(result,
                                  np.array([0, 0, 1, 0, 1, 0, 0]))



def test_check_spurious_simple() -> None:
    """Test that selection of runs of bad cycles are correct"""
    potentially_spurious = np.ones(50)
    potentially_spurious[30:] = 0

    result = diurnal.check_spurious(potentially_spurious)

    expected = np.zeros(50)
    expected[:30] = 1

    np.testing.assert_array_equal(result, expected)


def test_check_spurious_harder1() -> None:
    """Test that selection of runs of bad cycles are correct"""
    potentially_spurious = np.ones(50)
    potentially_spurious[40:] = 0
    potentially_spurious[np.array([10, 20, 30])] = 0

    result = diurnal.check_spurious(potentially_spurious)

    expected = np.zeros(50)
    expected[:40] = 1

    np.testing.assert_array_equal(result, expected)


def test_check_spurious_harder2() -> None:
    """Test that selection of runs of bad cycles are correct"""
    potentially_spurious = np.ones(50)
    potentially_spurious[40:] = 0
    potentially_spurious[np.array([10, 11])] = 0
    potentially_spurious[np.array([21, 22])] = diurnal.MISSING

    result = diurnal.check_spurious(potentially_spurious)

    expected = np.zeros(50)
    expected[:40] = 1

    np.testing.assert_array_equal(result, expected)


def test_check_spurious_harder3() -> None:
    """Test that selection of runs of bad cycles are correct"""
    potentially_spurious = np.ones(50)
    potentially_spurious[40:] = 0
    potentially_spurious[np.array([20, 21, 22])] = 0

    result = diurnal.check_spurious(potentially_spurious)

    expected = np.zeros(50)

    np.testing.assert_array_equal(result, expected)


@patch("diurnal.get_all_daily_offsets")
@patch("diurnal.get_potentially_spurious_days")
@patch("diurnal.check_spurious")
def test_diurnal_cycle_check(check_spurious_mock: Mock,
                             potentially_spurious_mock: Mock,
                             daily_offsets_mock: Mock) -> None:
    """Test calling and flag assignment"""
    # set up dummy returns
    daily_offsets_mock.return_value = (None, None)
    potentially_spurious_mock.return_value = None

    # set up data and dictionary
    ndata = 10
    indata = np.ma.arange(ndata*24)
    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    intimes = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                               for i in range(ndata*24)])[0])

    dummy_station = _setup_station(indata=indata, intimes=intimes)
    obs_var = dummy_station.temperature
    config_dict = {}
    config_dict[f"DIURNAL-{obs_var.name}"] = {"peak" : 10}

    # set up bad locs
    bad_locs = np.zeros(10)
    bad_locs[2] = 1
    check_spurious_mock.return_value = bad_locs

    diurnal.diurnal_cycle_check(obs_var, dummy_station, config_dict)

    expected = np.array(["" for i in range(obs_var.data.shape[0])])
    expected[48:72] = "u"

    np.testing.assert_array_equal(expected, obs_var.flags)



@patch("diurnal.diurnal_cycle_check")
def test_dcc(diurnal_checks_mock: Mock) -> None:

    temperatures = common.example_test_variable("temperature", np.arange(5))
    station = common.example_test_station(temperatures)

    diurnal.dcc(station, {})

    diurnal_checks_mock.assert_called_once()

    return