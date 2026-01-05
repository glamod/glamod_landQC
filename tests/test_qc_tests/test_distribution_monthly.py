"""
Contains tests for odd_cluster.py
"""
import numpy as np
import datetime as dt
import pandas as pd
from unittest.mock import patch, Mock

import distribution_monthly

import common
import utils

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


@patch("distribution_monthly.monthly_gap")
def test_dgc(monthly_gap_mock: Mock) -> None:
    """check driving routine"""
    station = _setup_station(np.ma.arange(10))

    # Do the call
    distribution_monthly.dgc(station, ["temperature"], {})

    # Mock to check call occurs as expected with right return
    monthly_gap_mock.assert_called_once()
