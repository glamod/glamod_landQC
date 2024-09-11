"""
Contains tests for spike.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock
import pytest

import spike
import qc_utils as utils

import common

# not testing plotting

def _set_up_data():
    series_length = 20
    values = np.ma.ones(series_length)
    values.mask = np.zeros(series_length)
    values.mask[0] = True

    times = np.ma.arange(series_length) * 60 # minutes
    times.mask = values.mask

    critical_values = {60: 5}

    return values, times, critical_values

def test_calculate_critical_values_short_record():

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

def test_calculate_critical_values():

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
    assert config_dict["SPIKE-temperature"] == {"60.0" : 6.5}

@pytest.mark.parametrize("name, expected",
                         [("temps", {60.0 : 1.0, 120.0: 2.0}),
                          ("dummy", {})])
def test_retreive_critical_values(name, expected):
    
    diffs = np.array([60.0, 120.0])
    config_dict = {"SPIKE-temps" : {"60.0" : "1.0", "120.0" : "2.0"}}

    values = spike.retreive_critical_values(diffs, config_dict, name)

    assert values == expected

@pytest.mark.parametrize("spike_points, expected", [([10], True),
                                                    ([10, 11], True),
                                                    ([10, 11, 12], True),
                                                    ([10, 11, 12, 13], False)])
def test_assess_potential_spike_single(spike_points, expected):
    
    values, times, critical_values = _set_up_data()
    values[spike_points] = 10
    value_diffs = np.ma.diff(values)
    time_diffs = np.diff(times)

    possible_spike, = np.nonzero(value_diffs > critical_values[60])

    is_spike, spike_len = spike.assess_potential_spike(time_diffs, value_diffs,
                                                         possible_spike[0], critical_values)

    assert is_spike == expected
    assert spike_len == len(spike_points)


@pytest.mark.parametrize("spike_points, spike_values, expected_is_spike",
                         [([10], [10], True),
                          ([10, 11], [10, 8], True),
                          ([10, 11], [10, 7], False),
                          ([10, 11], [10, 12], True),
                          ([10, 11], [10, 13], False),  # TODO: IS THIS CORRECT BEHAVIOUR
                          ([10, 11, 12], [10, 8, 10], True),
                          ([10, 11, 12], [10, 15, 10], False),  # TODO: IS THIS CORRECT BEHAVIOUR
                          ([10, 11, 12], [10, 8, 10], True),])
def test_assess_inside_spike(spike_points, spike_values, expected_is_spike):

    values, times, critical_values = _set_up_data()
    values[spike_points] = spike_values
    value_diffs = np.ma.diff(values)
    time_diffs = np.diff(times)

    possible_spike, = np.nonzero(value_diffs > critical_values[60])

    result_is_spike = spike.assess_inside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike

@pytest.mark.parametrize("spike_points, before_values, expected_is_spike",
                         [([10], [2, 3], True),
                          ([10, 11], [2, 3], True),
                          ([10], [0, 3], False), # diff > 5/2
                         ])
def test_assess_outside_spike_before(spike_points, before_values, expected_is_spike):

    values, times, critical_values = _set_up_data()
    values[spike_points] = 10
    values[spike_points[0]-len(before_values) : spike_points[0]] = before_values
    value_diffs = np.ma.diff(values)
    time_diffs = np.diff(times)
 
    possible_spike, = np.nonzero(value_diffs > critical_values[60])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike

@pytest.mark.parametrize("spike_points, before_values, expected_is_spike",
                         [([10], [3, 2], True),
                          ([10, 11], [3, 2], True),
                          ([10], [3, 0], False), # diff > 5/2
                         ])
def test_assess_outside_spike_affore(spike_points, before_values, expected_is_spike):

    values, times, critical_values = _set_up_data()
    values[spike_points] = 10
    values[spike_points[-1]+1: spike_points[-1]+1 + len(before_values)] = before_values
    value_diffs = np.ma.diff(values)
    time_diffs = np.diff(times)
 
    possible_spike, = np.nonzero(value_diffs > critical_values[60])

    result_is_spike = spike.assess_outside_spike(time_diffs, value_diffs,
                                         possible_spike[0], critical_values,
                                         True, len(spike_points))

    assert result_is_spike == expected_is_spike


def test_generate_differences():

    values = np.ma.ones(20)
    times = pd.Series([dt.datetime(2024, 1, 1, 12, 0) +
                          (i * dt.timedelta(seconds=60*60))
                          for i in range(len(values))])
    
    expected_value_diffs = np.ma.diff(values)
    expected_value_diffs.mask = np.zeros(expected_value_diffs.shape[0])

    expected_times_diffs = np.ma.diff(times)/np.timedelta64(1, "m")
    
    value_diffs, time_diffs, unique_diffs = spike.generate_differences(times, values)

    np.testing.assert_array_equal(value_diffs.data, expected_value_diffs.data)
    np.testing.assert_array_equal(time_diffs, expected_times_diffs)
    assert unique_diffs == np.array([60])

# def test_identify_spikes():

# def test_sc():