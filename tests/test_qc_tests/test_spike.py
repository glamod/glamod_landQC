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


def test_retreive_critical_values():
    
    diffs = np.array([60.0, 120.0])
    config_dict = {"SPIKE-temps" : {"60.0" : "1.0", "120.0" : "2.0"}}
    name = "temps"

    values = spike.retreive_critical_values(diffs, config_dict, name)

    assert values == {60.0 : 1.0, 120.0: 2.0}


# def test_assess_potential_spike():

# def test_assess_inside_spike():

# def test_assess_outside_spike():

# def test_identify_spikes():

# def test_sc():