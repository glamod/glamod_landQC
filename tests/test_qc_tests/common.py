"""
Contains common code for testing the QC tests
"""
import numpy as np
import datetime as dt
import pandas as pd

import utils

# Tests are called with a station, and some variables.
#   Need to build a station and empty variables to use

REPEATED_STREAK_STARTS_LENGTHS = {10: 3,
                                20: 3,
                                30: 3,
                                40: 3,
                                50: 3,
                                60: 3,
                                70: 4,
                                80: 4,
                                90: 4,
                                100: 4,
                                110: 4,
                                120: 5,
                                130: 5,
                                140: 5,
                                150: 6,
                                160: 6,
                                170: 7,
                                }

def example_test_variable(name: str,
                          vardata: np.ndarray,
                          mdi: float = -1.e30,
                          units: str = "degrees C",
                          dtype: str = "float") -> utils.MeteorologicalVariable:

    variable = utils.MeteorologicalVariable(name, mdi, units, dtype)

    variable.data = np.ma.masked_where(vardata == mdi, vardata)
    if len(variable.data.mask.shape) == 0:
        # single mask value, replace with array of True/False's
        if variable.data.mask:
            variable.data.mask = np.ones(variable.data.shape)
        else:
            variable.data.mask = np.zeros(variable.data.shape)

    variable.flags = np.array(["" for _ in vardata])

    return variable


def example_test_station(variable: utils.MeteorologicalVariable,
                         times: pd.Series | None = None,
                         latitude: int = 45,
                         longitude: int = 100,
                         elevation: int = 10) -> utils.Station:

    station = utils.Station("DummyID", latitude, longitude, elevation)

    setattr(station, variable.name, variable)

    if times is not None:
        # i.e. an array of times information has been supplied
        pass
    else:
        start_dt = dt.datetime(2000, 1, 1, 0, 0)
        times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                               for i in range(len(variable.data))])[0])

    station.set_times(times)
    station.set_datetime_values(np.array(times.dt.year),
                                np.array(times.dt.month),
                                np.array(times.dt.day),
                                np.array(times.dt.hour))

    return station


def generate_streaky_data(data: np.ma.MaskedArray,
                          starts_lengths: dict) -> np.ma.MaskedArray:
    """
    Using a dictionary of {start:length} pairs, make streaky data
    """

    for start, length in starts_lengths.items():
        data[start: start+length] = data[start]

    return data


def generate_station_for_clim_and_dist_tests(
                    varname: str = "temperature",
                    nyears: int = 10) -> utils.Station:
    """Create a station with temperatures (or other metric) for 10 Januaries
    :param str varname: name of variable to create
    :param str varname: name to call variable
    :param int nyears: number of years of data to generate
    Returns
    -------
    utils.Station
        Station object with data in temperature field
    """

    month_hours = 24*31
    # 10 years of Januaries
    indata = np.ma.array(np.tile(np.arange(24), 31*nyears))
    indata.mask = np.zeros(len(indata))

    # make MetVars
    obsvar = example_test_variable(varname, indata)

    for y in range(nyears):
        start_dt = dt.datetime(2000+y, 1, 1, 0, 0)
        month_times = pd.to_datetime(pd.DataFrame([start_dt + dt.timedelta(hours=i)\
                                for i in range(month_hours)])[0])
        if y == 0:
            times = month_times.copy()
        else:
            times = pd.concat([times, month_times])
    # check at this generation stage that have all Januaries
    assert np.unique(times.dt.month) == 1

    # make Station, by hand so can set times
    station = example_test_station(obsvar, times)

    return station