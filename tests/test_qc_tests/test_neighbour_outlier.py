"""
Contains tests for neighour_outlier.py
"""
import numpy as np
import datetime as dt
import pytest
from unittest.mock import patch, Mock

import neighbour_outlier
import qc_utils

import common


@patch("neighbour_outlier.neighbour_outlier")
def test_noc(neighbour_outlier_mock: Mock) -> None:

    # Set up data, variable & station
    temperatures = common.example_test_variable("temperature", np.arange(10))
    station = common.example_test_station(temperatures)

    # Do the call
    neighbour_outlier.noc(station, np.array(["ID", 20]), ["temperature"])

    # Mock to check call occurs as expected with right return
    neighbour_outlier_mock.assert_called_once()