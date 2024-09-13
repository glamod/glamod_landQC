"""
Contains common code for testing the QC tests
"""
import numpy as np
import datetime as dt

import qc_utils as utils

# Tests are called with a station, and some variables.
#   Need to build a station and empty variables to use

def example_test_variable(name,
                          vardata,
                          mdi=-1.e30,
                          units="degrees C",
                          dtype=(float)):

    variable = utils.Meteorological_Variable(name, mdi, units, dtype)

    variable.data = np.ma.masked_where(vardata == mdi, vardata)

    variable.flags = np.array(["" for i in range(len(vardata))])

    return variable


def example_test_station(variable,
                         latitude=45,
                         longitude=100,
                         elevation=10):

    station = utils.Station("DummyID", latitude, longitude, elevation)

    setattr(station, variable.name, variable)

    start_dt = dt.datetime(2000, 1, 1, 0, 0)
    station.times = np.array([start_dt + dt.timedelta(hours=i)\
                              for i in range(len(variable.data))])

    return station
