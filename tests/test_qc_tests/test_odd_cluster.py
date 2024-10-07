"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
from unittest.mock import patch, Mock
import pytest

import odd_cluster

import common
import qc_utils as utils



def _setup_station() -> utils.Station:

    # set up the data
    temps = np.ma.arange(10)
    temps.mask = np.zeros(len(temps))

    # make MetVars
    temperature = common.example_test_variable("temperature", temps)
    
    # make Station
    station = common.example_test_station(temperature)

    return station


@patch("odd_cluster.flag_clusters")
def test_read_hcc(flag_clusters_mock: Mock) -> None:

    station = _setup_station()

    # Do the call
    odd_cluster.occ(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    flag_clusters_mock.assert_called_once()
