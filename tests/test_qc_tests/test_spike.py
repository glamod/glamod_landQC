"""
Contains tests for spike.py
"""
import numpy as np
import datetime as dt
import pandas as pd
import pytest
from unittest.mock import patch, Mock

import spike

import common

# not testing plotting

def _set_up_data() -> tuple[np.array, np.array, dict]:
    series_length = 20
    values = np.ma.ones(series_length)
    values.mask = np.zeros(series_length)

    times = np.ma.arange(series_length) * 60 # minutes
    times.mask = values.mask

    critical_values = {60.: 5}

    return values, times, critical_values

def _set_up_masked_data() -> tuple[np.array, np.array, dict]:
    series_length = 20
    values = np.ma.ones(series_length)
    values.mask = np.zeros(series_length)
    values.mask[9] = True
    values.mask[12] = True

    times = np.ma.arange(series_length) * 60 # minutes
    times.mask = values.mask

    critical_values = {60.: 5, 120.: 5}

    return values, times, critical_values


def test_calculate_critical_values_short_record() -> None:

    # set up the data
    temps = np.ma.arange(10)
    temps.mask = np.zeros(10)
    temps.mask[2:4] = 1
    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(temps))])

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)

    config_dict = {}
    spike.calculate_critical_values(temperature, times, config_dict)

    # insufficient data, no critical value stored for this time difference
    assert config_dict == {}


def test_calculate_critical_values() -> None:

    # set up the data
    temps = np.ma.ones(200) * 10
    temps[::5] = 11
    temps[::10] = 12
    temps[::20] = 14
    temps[::50] = 16

    temps.mask = np.zeros(200)
    temps.mask[2:4] = 1
    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(temps))])

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)

    config_dict = {}
    spike.calculate_critical_values(temperature, times, config_dict, plots=True)

    # qc_utils routine to be tested elsewhere
    assert config_dict["SPIKE-temperature"] == {60. : 6.5}


@pytest.mark.parametrize("name, expected",
                         [("temps", {60. : 1.0, 120.: 2.0}),
                          ("dummy", {})])
def test_retreive_critical_values(name: str, expected: dict) -> None:
    
    diffs = np.array([60., 120.])
    config_dict = {"SPIKE-temps" : {60. : 1.0, 120. : 2.0}}

    values = spike.retreive_critical_values(diffs, config_dict, name)

    assert values == expected


@pytest.mark.parametrize("spike_points, expected", [(np.array([10]), True),
                                                    (np.array([10, 11]), True),
                                                    (np.array([10, 11, 12]), True),
                                                    (np.array([10, 11, 12, 13]), False)])
def test_assess_potential_spike_single(spike_points: np.array,
                                       expected: bool) -> None:
    
    values, times, critical_values = _set_up_data()
    values.data[spike_points] = 10

    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    is_spike, spike_len = spike.assess_potential_spike(time_diffs, value_diffs,
                                                         possible_spike[0], critical_values)

    assert is_spike == expected
    assert spike_len == len(spike_points)


@pytest.mark.parametrize("spike_points, expected", [(np.arange(10, 11), True),
                                                    (np.arange(10, 12), True),
                                                    (np.arange(10, 13), True),
                                                    (np.arange(10, 14), True),
                                                    (np.arange(10, 15), False)])
# point 12 masked, so 3[4] point spike in the penultimate[final] case
def test_assess_potential_spike_single_masked(spike_points: np.array,
                                              expected: bool) -> None:
    
    values, times, critical_values = _set_up_masked_data()
    values.data[spike_points] = 10

    non_mask_locs, = np.nonzero(values.mask == False)
    expected_length = len([sp for sp in spike_points if sp in non_mask_locs])

    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    is_spike, spike_len = spike.assess_potential_spike(time_diffs, value_diffs,
                                                         possible_spike[0], critical_values)

    assert is_spike == expected
    assert spike_len == expected_length


@pytest.mark.parametrize("spike_points, spike_values, expected_is_spike",
                         [(np.array([10]), np.array([10]), True),
                          (np.array([10, 11]), np.array([10, 8]), True),
                          (np.array([10, 11]), np.array([10, 7]), False),
                          (np.array([10, 11]), np.array([10, 12]), True),
                          (np.array([10, 11]), np.array([10, 13]), False),  # TODO: IS THIS CORRECT BEHAVIOUR
                          (np.array([10, 11, 12]), np.array([10, 8, 10]), True),
                          (np.array([10, 11, 12]), np.array([10, 15, 10]), False),  # TODO: IS THIS CORRECT BEHAVIOUR
                          (np.array([10, 11, 12]), np.array([10, 8, 10]), True),])
def test_assess_inside_spike(spike_points: np.array,
                             spike_values: np.array,
                             expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_data()
    values[spike_points] = spike_values
    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_inside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


@pytest.mark.parametrize("spike_points, spike_values, expected_is_spike",
                         [(np.array([10, 11, 12]), np.array([10, 8, 0]), True), # 12 masked so no impact
                          (np.array([10, 11, 12]), np.array([10, 8, 10]), True),
                          (np.array([10, 11, 12, 13]), np.array([10, 8, 8, 8]), True),
                          (np.array([10, 11, 12, 13]), np.array([10, 8, 6, 4]), False),
                         ])
def test_assess_inside_spike_masked(spike_points: np.array,
                                    spike_values: np.array,
                                    expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_masked_data()
    values.data[spike_points] = spike_values

    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    # retain only non-masked indices in spike_points
    non_mask_locs, = np.nonzero(values.mask == False)
    spike_points = np.array([sp for sp in spike_points if sp in non_mask_locs])

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_inside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike
    

@pytest.mark.parametrize("spike_points, before_values, expected_is_spike",
                         [(np.array([10]), np.array([2, 3]), True),
                          (np.array([10, 11]), np.array([2, 3]), True),
                          (np.array([10]), np.array([0, 3]), False), # diff from 0 to 3 > 5/2
                         ])
def test_assess_outside_spike_before(spike_points: np.array,
                                     before_values: np.array,
                                     expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_data()
    values.data[spike_points] = 10
    values.data[spike_points[0]-len(before_values) : spike_points[0]] = before_values
    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())
 
    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


@pytest.mark.parametrize("spike_points, before_values, expected_is_spike",
                         [(np.array([10]), np.array([2, 3]), True), # diff < 5/2,
                          (np.array([10, 11]), np.array([2, 3]), True), # diff < 5/2,
                          (np.array([10]), np.array([0, 3]), True), # diff from 0 to 3 > 5/2, but 3 masked, so not counted
                          (np.array([10, 11]), np.array([5, 2, 1]), False), # diff from 5 to 2 > 5/2, as 1 masked
                         ])
def test_assess_outside_spike_before_masked(spike_points: np.array,
                                            before_values: np.array,
                                            expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_masked_data()
    # 9 and 12 masked
    values.data[spike_points] = 10
    values.data[spike_points[0]-len(before_values) : spike_points[0]] = before_values

    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


@pytest.mark.parametrize("spike_points, after_values, expected_is_spike",
                         [(np.array([10]), np.array([3, 2]), True),
                          (np.array([10, 11]), np.array([3, 2]), True),
                          (np.array([10]), np.array([3, 0]), False), # diff > 5/2
                         ])
def test_assess_outside_spike_after(spike_points: np.array,
                                    after_values: np.array,
                                    expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_data()
    values.data[spike_points] = 10
    values.data[spike_points[-1]+1: spike_points[-1]+1 + len(after_values)] = after_values
    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())
 
    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


@pytest.mark.parametrize("spike_points, after_values, expected_is_spike",
                         [(np.array([10]), np.array([3, 2]), True),
                          (np.array([10, 11]), np.array([3, 2]), True), # diff from 2 to 1 < 5/2, as 3 masked
                          (np.array([10, 11]), np.array([3, 2, 5]), False), # diff from 5 to 2 > 5/2, as 3 masked
                         ])
def test_assess_outside_spike_after_masked(spike_points: np.array,
                                           after_values: np.array,
                                           expected_is_spike: bool) -> None:

    values, times, critical_values = _set_up_masked_data()
    # 9 and 12 masked
    values.data[spike_points] = 10
    values.data[spike_points[-1]+1: spike_points[-1]+1 + len(after_values)] = after_values

    value_diffs = np.ma.diff(values.compressed())
    time_diffs = np.diff(times.compressed())

    possible_spike, = np.nonzero(value_diffs > critical_values[60.])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


def test_generate_differences() -> None:

    values = np.ma.ones(20)
    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(values))])
    
    expected_value_diffs = np.ma.zeros(19)
    expected_value_diffs.mask = np.zeros(expected_value_diffs.shape[0])

    expected_times_diffs = np.ma.diff(times)/np.timedelta64(1, "m")
    
    value_diffs, time_diffs, unique_diffs = spike.generate_differences(times, values)

    np.testing.assert_array_equal(value_diffs.data, expected_value_diffs.data)
    np.testing.assert_array_equal(time_diffs, expected_times_diffs)
    assert unique_diffs == np.array([60])


def test_generate_masked_differences() -> None:

    values = np.ma.ones(20)
    values[::5] = 2
    values.mask = np.zeros(values.shape[0])
    values.mask[::4] = True

    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(values))])
    
    # more complicated pattern of changes, so insert manually
    expected_value_diffs = np.ma.zeros(14)
    expected_value_diffs[2::4] = 1
    expected_value_diffs[3::4] = -1
    expected_value_diffs.mask = np.zeros(expected_value_diffs.shape[0])

    expected_times_diffs = np.ma.diff(times[:15])/np.timedelta64(1, "m")
    expected_times_diffs[2::3] *= 2

    value_diffs, time_diffs, unique_diffs = spike.generate_differences(times, values)

    np.testing.assert_array_equal(value_diffs.data, expected_value_diffs.data)
    np.testing.assert_array_equal(time_diffs, expected_times_diffs)
    # have two time differences, as masked values skipped over
    np.testing.assert_array_equal(unique_diffs, np.array([60, 120]))


@pytest.mark.parametrize("spike_points",
                         [(np.array([10])),
                          (np.array([10, 15])),
                          (np.array([10, 11])),
                          (np.array([10, 11, 12])),
                          (np.array([10, 11, 12, 13])),
                          (np.array([10, 11, 12, 13])),
                         ])
def test_identify_spikes(spike_points: np.array) -> None:

    values, times, critical_values = _set_up_masked_data()
    # 9 & 12 masked
    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(values))])

    # make the spike
    values.data[spike_points] = 10
    flags = np.array(["" for _ in values])
    spike_flags = np.copy(flags)
    
    # adjust the locations which will be flagged using the data mask
    non_mask_locs, = np.nonzero(values.mask == False)
    spike_points = np.array([sp for sp in spike_points if sp in non_mask_locs])
    spike_flags[spike_points] = "S"
    
    # generate the example variable
    obs_var = common.example_test_variable("temperature", values)
    obs_var.flags = flags

    # create the config dictionary
    config_dict = {}
    config_dict["SPIKE-{}".format(obs_var.name)]  = critical_values

    # run the routine
    spike.identify_spikes(obs_var, times, config_dict)

    #check the flags set correctly
    np.testing.assert_array_equal(obs_var.flags, spike_flags)
    assert obs_var.flags[spike_points[0]] == "S"


@pytest.mark.parametrize("full", [True, False])
@patch("spike.identify_spikes")
@patch("spike.calculate_critical_values")
def test_sc(critical_values_mock: Mock,
            identify_spikes_mock: Mock, 
            full: bool):
    
    var = common.example_test_variable("dummy", np.ones(10))
    station = common.example_test_station(var)

    spike.sc(station, ["dummy"], {}, full=full)

    if full:
        critical_values_mock.assert_called_once()
    identify_spikes_mock.assert_called_once()