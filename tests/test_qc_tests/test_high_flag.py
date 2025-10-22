"""
Contains tests for high_flag.py
"""
import numpy as np
import pytest
from unittest.mock import patch, Mock

import high_flag
import utils

import common


def test_set_synergistic_flags_short_record() -> None:

    slp = common.example_test_variable("sea_level_pressure",
                                       np.arange(10))
    slp.flags[:] = "H"

    stnlp = common.example_test_variable("station_level_pressure",
                                         np.arange(10))

    station = common.example_test_station(slp)
    station.station_level_pressure = stnlp

    high_flag.set_synergistic_flags(station, "station_level_pressure")
    # no flags set as short record
    expected_flags = np.array(["" for _ in range(stnlp.data.shape[0])])

    np.testing.assert_array_equal(station.station_level_pressure.flags, expected_flags)


def test_set_synergistic_flags() -> None:

    slp = common.example_test_variable("sea_level_pressure",
                                       np.arange(11*utils.DATA_COUNT_THRESHOLD))
    slp.flags[:] = "H"

    stnlp = common.example_test_variable("station_level_pressure",
                                         np.arange(11*utils.DATA_COUNT_THRESHOLD))

    station = common.example_test_station(slp)
    station.station_level_pressure = stnlp

    high_flag.set_synergistic_flags(station, "station_level_pressure")

    # flag all values in the record for this part of this test
    expected_flags = np.array(["H" for _ in range(stnlp.data.shape[0])])

    np.testing.assert_array_equal(station.station_level_pressure.flags, expected_flags)


def test_high_flag_rate_noflags() -> None:

    temperatures = common.example_test_variable("temperature",
                                                np.arange(11*utils.DATA_COUNT_THRESHOLD))

    new_flags, flags_set = high_flag.high_flag_rate(temperatures)
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])

    assert flags_set is False
    np.testing.assert_array_equal(new_flags, expected_flags)


def test_high_flag_rate_short_record() -> None:

    temperatures = common.example_test_variable("temperature", np.arange(5))

    new_flags, flags_set = high_flag.high_flag_rate(temperatures)

    assert flags_set is False
    np.testing.assert_array_equal(new_flags, np.array(["" for _ in range(5)]))


def test_high_flag_rate_prior_run() -> None:

    temperatures = common.example_test_variable("temperature",
                                                np.arange(11*utils.DATA_COUNT_THRESHOLD))
    temperatures.flags[:10] = "H"
    # Set a large number of other flags, so should trigger check
    temperatures.flags[2*utils.DATA_COUNT_THRESHOLD:] = "L"
    new_flags, flags_set = high_flag.high_flag_rate(temperatures)
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])

    assert flags_set is False
    np.testing.assert_array_equal(new_flags, expected_flags)


def test_high_flag_rate_low_fraction() -> None:

    temperatures = common.example_test_variable("temperature",
                                                np.arange(11*utils.DATA_COUNT_THRESHOLD))

    # set many flags but too few to trigger the test
    temperatures.flags[:int(temperatures.data.shape[0]*utils.HIGH_FLAGGING) -1] = "L"
    new_flags, flags_set = high_flag.high_flag_rate(temperatures)
    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])

    assert flags_set is False
    np.testing.assert_array_equal(new_flags, expected_flags)


def test_high_flag_rate_high_fraction() -> None:

    temperatures = common.example_test_variable("temperature",
                                                np.arange(11*utils.DATA_COUNT_THRESHOLD))

    # set many flags to trigger the test
    temperatures.flags[:int(temperatures.data.shape[0]*utils.HIGH_FLAGGING) + 1] = "L"

    expected_flags = np.array(["" for _ in range(temperatures.data.shape[0])])
    expected_flags[int(temperatures.data.shape[0]*utils.HIGH_FLAGGING) + 1:] = "H"

    new_flags, flags_set = high_flag.high_flag_rate(temperatures)

    assert flags_set is True
    np.testing.assert_array_equal(new_flags, expected_flags)


@patch("high_flag.high_flag_rate")
def test_pcc(high_flag_rate_mock: Mock) -> None:

    temperatures = common.example_test_variable("temperature", np.arange(5))
    station = common.example_test_station(temperatures)

    expected_flags = np.array(["H" for _ in range(temperatures.data.shape[0])])
    high_flag_rate_mock.return_value = (expected_flags, True)

    set_vars = high_flag.hfr(station, ["temperature"])

    # Mock to check call occurs as expected with right return
    high_flag_rate_mock.assert_called_once()

    np.testing.assert_array_equal(temperatures.flags,
                                  expected_flags)

    assert set_vars == 1


@pytest.mark.parametrize("name", ("sea_level_pressure",
                                  "station_level_pressure",
                                  "wind_speed",
                                  "wind_direction"))
@patch("high_flag.set_synergistic_flags")
@patch("high_flag.high_flag_rate")
def test_pcc_synergistic(high_flag_rate_mock: Mock,
                         synergistic_flags_mock: Mock,
                         name: str) -> None:

    var = common.example_test_variable(name, np.arange(5))
    station = common.example_test_station(var)

    expected_flags = np.array(["H" for _ in range(var.data.shape[0])])
    high_flag_rate_mock.return_value = (expected_flags, True)

    set_vars = high_flag.hfr(station, [name])

    # Mock to check call occurs as expected with right return
    high_flag_rate_mock.assert_called_once()

    np.testing.assert_array_equal(var.flags,
                                  expected_flags)

    synergistic_flags_mock.assert_called_once()
    # See comment in code, synergistically flagged ones only count once
    assert set_vars == 1
