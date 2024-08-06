"""
Contains common code for testing the QC tests
"""
import numpy as np

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
    variable.flags = np.array(["" for _ in vardata])

    return variable


def example_test_station(variable,
                         latitude=45,
                         longitude=100,
                         elevation=10):

    station = utils.Station("DummyID", latitude, longitude, elevation)

    setattr(station, variable.name, variable)

    return station
