"""
Wind Cross Checks
^^^^^^^^^^^^^^^^^^^^^

Cross checks on speed and direction.
"""
import numpy as np

import qc_utils as utils
#************************************************************************

# TODO - Add wind rose check if deemed robust enough

#************************************************************************

#************************************************************************
def logical_checks(speed, direction, fix=False, plots=False, diagnostics=False):
    """
    Select occurrences of wind speed and direction which are 
    logically inconsistent with measuring practices.

    From Table 2 - DeGaetano, JOAT, 14, 308-317, 1997

    :param MetVar speed: wind speed object
    :param MetVar direction: wind direction object
    :param bool fix: fix the zero speed no direction entries [False]
    :param bool plots: do plots?
    :param bool diagnostics: do diagnostics?

    """
    sflags = np.array(["" for i in range(speed.data.shape[0])])
    dflags = np.array(["" for i in range(speed.data.shape[0])])

    # recover direction information where the speed is Zero
    fix_zero_direction = np.ma.where(np.logical_and(speed.data == 0, direction.data.mask == True))
    if fix:
        direction.data[fix_zero_direction] = 0
        direction.data.mask[fix_zero_direction] = False
    else:
        dflags[fix_zero_direction] = "w"
    if diagnostics:
        print("  Zero direction : {}".format(len(fix_zero_direction[0])))

    # negative speeds (can't fix)
    negative_speed = np.ma.where(speed.data < 0)
    sflags[negative_speed] = "w"
    if diagnostics:
        print("  Negative speed : {}".format(len(negative_speed[0])))

    # negative directions (don't try to adjust)
    negative_direction = np.ma.where(direction.data < 0)
    dflags[negative_direction] = "w"
    if diagnostics:
        print("  Negative direction : {}".format(len(negative_direction[0])))

    # wrapped directions (don't try to adjust)
    wrapped_direction = np.ma.where(direction.data > 360)
    dflags[wrapped_direction] = "w"
    if diagnostics:
        print("  Wrapped direction : {}".format(len(wrapped_direction[0])))

    # no direction possible if speed == 0
    bad_direction = np.ma.where(np.logical_and(speed.data == 0, direction.data != 0))
    dflags[bad_direction] = "w"
    if diagnostics:
        print("  Bad direction : {}".format(len(bad_direction[0])))

    # northerlies given as 360, not 0 --> calm
    bad_speed = np.ma.where(np.logical_and(direction.data == 0, speed.data != 0))
    sflags[bad_speed] = "w"
    if diagnostics:
        print("  Bad speed : {}".format(len(bad_speed[0])))

    # copy flags into attribute
    speed.flags = utils.insert_flags(speed.flags, sflags)
    direction.flags = utils.insert_flags(direction.flags, dflags)

    if diagnostics:

        print("Wind Logical".format(speed.name))
        print("   Cumulative number of {} flags set: {}".format(speed.name, len(np.where(sflags != "")[0])))
        print("   Cumulative number of {} flags set: {}".format(direction.name, len(np.where(dflags != "")[0])))

    return # logical_checks

#************************************************************************
def wcc(station, config_dict, fix=False, full=False, plots=False, diagnostics=False):
    """
    Extract the variables and pass to the Wind Cross Checks

    :param Station station: Station Object for the station
    :param str config_dict: string for configuration file (unused here)
    :param bool fix: repair/amend values as a result of logical checks [False]
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    speed = getattr(station, "wind_speed")
    direction = getattr(station, "wind_direction")

    logical_checks(speed, direction, fix=fix, plots=plots, diagnostics=diagnostics)

    return # pcc

#************************************************************************
if __name__ == "__main__":

    print("wind cross checks")
#************************************************************************
