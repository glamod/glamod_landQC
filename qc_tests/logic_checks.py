"""
Logic Checks
^^^^^^^^^^^^

Check for illogical values
"""
#************************************************************************
import os
import json
import datetime as dt
import numpy as np
import setup

import qc_utils as utils

with open(utils.LOGICFILE, "r") as lf:
    REASONABLE_LIMITS = json.load(lf)["logic_limits"]

#************************************************************************
def logic_check(obs_var, plots=False, diagnostics=False):
    """
    Check for exceedences of world record values

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    obs_min, obs_max = REASONABLE_LIMITS[obs_var.name]

    bad_locs, = np.where(np.logical_and(obs_var.data < obs_min, obs_var.data > obs_max))

    if len(bad_locs) > 0:
        flags[bad_locs] = "L"

        if diagnostics:
            print("Logic Checks {}".format(obs_var.name))
            print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return flags # logic_check 

#************************************************************************
def write_logic_error(station, message, diagnostics=False):
    """
    Write a logic error at the station metadata level to file

    :param Station station: met. station.
    :param str message: the string to write into the file
    :param bool diagnostics: turn on diagnostic output
    """

    outfilename = os.path.join(setup.SUBDAILY_ERROR_DIR, "{}.err".format(station.id))

    with open(outfilename, "a") as outfile:
        
        outfile.write(message + "\n")

    return # write_logic_error

#************************************************************************
def lc(station, var_list, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Logic Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    # https://github.com/glamod/glamod-dm/blob/master/glamod-parser/glamod/parser/filters/observations_table.py
    # database parser has these, for future reference

    # station level
    return_code = 0
    if station.lat < -90 or station.lat > 90:
        write_logic_error(station, "Bad latitude: {}".format(station.lat), diagnostics=diagnostics)
        if diagnostics:
            print("Bad latitude: {}".format(station.lat))
        return_code = -1

    if station.lon < -180 or station.lon > 180:
        write_logic_error(station, "Bad longtitude: {}".format(station.lon), diagnostics=diagnostics)
        if diagnostics:
            print("Bad longtitude: {}".format(station.lon))
        return_code = -1

    if station.lon == 0 and station.lat == 0:
        write_logic_error(station, "Bad longtitude & latitude combination: lon={}, lat={}".format(station.lon, station.lat), diagnostics=diagnostics)
        if diagnostics:
            print("Bad longtitude/latitude: {} & {}".format(station.lon, station.lat))
        return_code = -1

    if station.elev < -432.65 or station.elev > 8850:
        write_logic_error(station, "Bad elevation: {}".format(station.elev), diagnostics=diagnostics)
        if diagnostics:
            print("Bad elevation: {}".format(station.elev))
        return_code = -1

    if station.times.iloc[0] < dt.datetime(1650,1,1):
        write_logic_error(station, "Bad start time: {}".format(station.times[0]), diagnostics=diagnostics)
        if diagnostics:
            print("Bad start time: {}".format(station.times[0]))
        return_code = -1

    elif station.times.iloc[-1] > dt.datetime.now():
        write_logic_error(station, "Bad end time: {}".format(station.times[-1]), diagnostics=diagnostics)
        if diagnostics:
            print("Bad end time: {}".format(station.times[-1]))
        return_code = -1
    
    # observation level
    for var in var_list:

        obs_var = getattr(station, var)

        flags = logic_check(obs_var, plots=plots, diagnostics=diagnostics)

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    return return_code # lc

#************************************************************************
if __name__ == "__main__":

    print("checking for illogical values")
#************************************************************************
