"""
Contains common code for testing the QC tests
"""
import numpy as np

import qc_utils as utils

# Tests are called with a station, and some variables.
#   Need to build a station and empty variables to use

def example_test_variable(name: str,
                          vardata: np.array,
                          mdi: float = -1.e30,
                          units: str = "degrees C",
                          dtype: tuple = (float)) -> utils.Meteorological_Variable:

    variable = utils.Meteorological_Variable(name, mdi, units, dtype)

    variable.data = np.ma.masked_where(vardata == mdi, vardata)

    variable.flags = np.array(["" for i in range(len(vardata))])

    return variable


def example_test_station(variable: utils.Meteorological_Variable,
                         latitude: int = 45,
                         longitude: int = 100,
                         elevation: int = 10) -> utils.Station:

    station = utils.Station("DummyID", latitude, longitude, elevation)

    setattr(station, variable.name, variable)

    return station


def add_times_to_example_station(station: utils.Station,
                                 times: np.array) -> None:
    
    station.years = np.array([d.year for d in times])
    station.months = np.array([d.month for d in times])
    station.days = np.array([d.day for d in times])
    station.hours = np.array([d.hour for d in times])
    station.times = times

