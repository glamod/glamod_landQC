"""
Contains tests for timestamp.py
"""
import numpy as np
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import timestamp
import qc_utils

import common

# not testing plotting

# def test_identify_multiple_values()


@patch("timestamp.identify_multiple_values")
def test_tsc(multiple_values_mock: Mock) -> None:

    # Set up data, variable & station
    obs_var = common.example_test_variable("temperature", np.ones(10))
    station = common.example_test_station(obs_var)

    # Set up flags to uses mocked return
    multiple_values_mock.return_value = True

    # Do the call
    timestamp.tsc(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    multiple_values_mock.assert_called_once()