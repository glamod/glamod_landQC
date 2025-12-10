"""
Logic Checks
============

Check for illogical values as outlined in json file
"""
#************************************************************************
import json
import datetime as dt
import numpy as np
import pandas as pd
import setup
import logging
logger = logging.getLogger(__name__)

import utils
#************************************************************************

BAD_THRESHOLD = 0.005 # 99.5% good, 0.5% bad

with open(utils.LOGICFILE, "r") as lf:
    REASONABLE_LIMITS = json.load(lf)["logic_limits"]

#************************************************************************
def logic_check(obs_var: utils.MeteorologicalVariable, plots: bool = False,
                diagnostics: bool = False) -> np.ndarray:
    """
    Check for exceedences of world record values

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: array of flags
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    obs_min, obs_max = REASONABLE_LIMITS[obs_var.name]

    bad_locs, = np.ma.where(np.logical_or(obs_var.data < obs_min, obs_var.data > obs_max))

    if len(bad_locs) > 0:

        # are more than a certain fraction bad
        #    (pervasive issues only as limits are within World Records)

        if (bad_locs.shape[0] / obs_var.data.compressed().shape[0]) > BAD_THRESHOLD: # 99.5% good, 0.5% bad

            flags[bad_locs] = "L"
            logger.info(f"Logic Checks {obs_var.name}")
            logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")
        else:
            logger.info(f"Logic Checks {obs_var.name}")
            logger.info(f"   Number of issues found: {len(bad_locs)}")
            logger.info("   No flags set as proportion small enough")

    return flags # logic_check

#************************************************************************
def write_logic_error(station: utils.Station, message: str, diagnostics: bool = False) -> None:
    """
    Write a logic error at the station metadata level to file

    :param Station station: met. station.
    :param str message: the string to write into the file
    :param bool diagnostics: turn on diagnostic output
    """

    outfilename = setup.SUBDAILY_ERROR_DIR / f"{station.id}.err"

    with open(outfilename, "a") as outfile:
        outfile.write(dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M") + "\n")
        outfile.write(message + "\n")

    # write_logic_error


#************************************************************************
def lc(station: utils.Station, var_list: list, full: bool = False,
       plots: bool = False, diagnostics: bool = False) -> int:
    """
    Run through the variables and pass to the Logic Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: return_code, int of 0 (pass) or -1 (fail)
    """
    # https://github.com/glamod/glamod-dm/blob/master/glamod-parser/glamod/parser/filters/observations_table.py
    # database parser has these, for future reference

    # station level (from inventory listing, not for each timestamp)
    return_code = 0
    if station.lat < -90 or station.lat > 90:
        write_logic_error(station, f"Bad latitude: {station.lat}", diagnostics=diagnostics)
        logger.warning(f"Bad latitude: {station.lat}")
        return_code = -1

    if station.lon < -180 or station.lon > 180:
        write_logic_error(station, f"Bad longtitude: {station.lon}", diagnostics=diagnostics)
        logger.warning(f"Bad longitude: {station.lon}")
        return_code = -1

    if station.lon == 0 and station.lat == 0:
        write_logic_error(station,
                          f"Bad longtitude & latitude combination: lon={station.lon}, lat={station.lat}",
                          diagnostics=diagnostics)
        logger.warning(f"Bad longitude/latitude combination: {station.lon} & {station.lat}")
        return_code = -1

    # Missing elevation acceptable - removed this for the moment (7 November 2019, RJHD)
    #       missing could be -999, -999.9, -999.999 or even 9999.0 etc hence using string comparison
    if (station.elev < -432.65 or station.elev > 8850.):
        if str(station.elev)[:4] not in utils.ALLOWED_MISSING_ELEVATIONS:
            write_logic_error(station, f"Bad elevation: {station.elev}", diagnostics=diagnostics)
            logger.warning(f"Bad elevation: {station.elev}")
            return_code = -1
        else:
            logger.warning(f"Elevation missing (but not flagged): {station.elev}")

    if station.times.iloc[0] < dt.datetime(1700, 1, 1):
        # Pandas datetime limited to pd.Timestamp.min = Timestamp('1677-09-22 00:12:43.145225')
        write_logic_error(station, f"Bad start time: {station.times[0]}", diagnostics=diagnostics)
        logger.warning(f"Bad start time: {station.times[0]}")
        return_code = -1

    elif station.times.iloc[-1] > dt.datetime.now():
        # Pandas datetime limited to pd.Timestamp.max = Timestamp('2262-04-11 23:47:16.854775807')
        write_logic_error(station, f"Bad end time: {station.times[-1]}", diagnostics=diagnostics)
        logger.warning(f"Bad end time: {station.times[-1]}")
        return_code = -1

    # For Release 7 had some discontinuities in the times, with a repeated chunk
    #  Time stamp went from 2023/5/31/21:00 to 2023/1/1/00:00
    time_differences = np.diff(station.times.dt.to_pydatetime())
    bad_differences, = np.nonzero(time_differences < dt.timedelta(0))
    if len(bad_differences) != 0:
        for location in bad_differences:
            write_logic_error(station,
                              f"Dates not in ascending order:\n{station.times[location: location+2].to_string()}",
                              diagnostics=diagnostics)
            logger.warning(f"Dates not in ascending order:\n{station.times[location: location+2].to_string()}")

        return_code = -1

    # Now do logic checks on observations

    # observation level
    for var in var_list:

        obs_var = getattr(station, var)

        flags = logic_check(obs_var, plots=plots, diagnostics=diagnostics)

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    return return_code # lc

