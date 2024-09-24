"""
Contains common code for testing the QC tests
"""
import numpy as np
import datetime as dt
import pandas as pd

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
    variable.flags = np.array(["" for _ in vardata])

    variable.flags = np.array(["" for i in range(len(vardata))])

    return variable


def example_test_station(variable: utils.Meteorological_Variable,
                         latitude: int = 45,
                         longitude: int = 100,
                         elevation: int = 10) -> utils.Station:

    station = utils.Station("DummyID", latitude, longitude, elevation)

    setattr(station, variable.name, variable)

    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                              for i in range(len(variable.data))])[0])
    
    station.times = times
    station.years = np.array(times.dt.year)
    station.months = np.array(times.dt.month)
    station.days = np.array(times.dt.day)
    station.hours = np.array(times.dt.hour)
    
    return station

