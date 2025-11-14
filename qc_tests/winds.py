"""
Wind Cross Checks
=================

Cross checks on speed and direction.
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils
#************************************************************************

# TODO - Add wind rose check if deemed robust enough

#************************************************************************

#************************************************************************
def logical_checks(speed: utils.MeteorologicalVariable,
                   direction: utils.MeteorologicalVariable,
                   fix: bool = False, plots: bool = False,
                   diagnostics: bool = False) -> np.ndarray:
    """
    Select occurrences of wind speed and direction which are
    logically inconsistent with measuring practices.

    From Table 2 - DeGaetano, JOAT, 14, 308-317, 1997

    :param MetVar speed: wind speed object
    :param MetVar direction: wind direction object
    :param bool fix: fix the zero speed no direction entries [False]
    :param bool plots: do plots?
    :param bool diagnostics: do diagnostics?

    :returns: Array of locations where direction data have been corrected
    """
    sflags = np.array(["" for i in range(speed.data.shape[0])])
    dflags = np.array(["" for i in range(speed.data.shape[0])])

    # recover direction information where the speed is Zero
    fix_zero_direction, = np.ma.where(np.logical_and(speed.data == 0,
                                                     direction.data.mask == True))
    if fix:
        direction.data[fix_zero_direction] = 0
        direction.data.mask[fix_zero_direction] = False
        if diagnostics:
            print("  Zero direction : {}".format(len(fix_zero_direction)))
    else:
        dflags[fix_zero_direction] = "1"
        if diagnostics:
            print("  Zero direction : {}".format(len(fix_zero_direction)))
        # and set to empty as can be used in parent to copy values to dataframe
        fix_zero_direction = np.array([])

    # negative speeds (can't fix)
    negative_speed = np.ma.where(speed.data < 0)
    sflags[negative_speed] = "w"
    logger.info(f"  Negative speed : {len(negative_speed[0])}")

    # negative directions (don't try to adjust)
    negative_direction = np.ma.where(direction.data < 0)
    dflags[negative_direction] = "w"
    logger.info(f"  Negative direction : {len(negative_direction[0])}")

    # wrapped directions (don't try to adjust)
    wrapped_direction = np.ma.where(direction.data > 360)
    dflags[wrapped_direction] = "w"
    logger.info(f"  Wrapped direction : {len(wrapped_direction[0])}")

    # no direction possible if speed == 0
    bad_direction = np.ma.where(np.logical_and(speed.data == 0,
                                               direction.data != 0))
    dflags[bad_direction] = "w"
    logger.info(f"  Bad direction : {len(bad_direction[0])}")

    # northerlies given as 360, not 0 --> calm
    bad_speed = np.ma.where(np.logical_and(direction.data == 0, speed.data != 0))
    sflags[bad_speed] = "w"
    logger.info(f"  Bad speed : {len(bad_speed[0])}")

    # copy flags into attribute
    speed.store_flags(utils.insert_flags(speed.flags, sflags))
    direction.store_flags(utils.insert_flags(direction.flags, dflags))

    if diagnostics:

        print("Wind Logical")
        print(f"   Cumulative number of {speed.name} flags set: {np.count_nonzero(sflags != '')}")
        print(f"   Cumulative number of {direction.name} flags set: {np.count_nonzero(dflags == 'w')}")
        print(f"   Cumulative number of {direction.name} convention flags set: {np.count_nonzero(dflags == '1')}")

    logger.info("Wind Logical")
    logger.info(f"   Cumulative number of {speed.name} flags set: {np.count_nonzero(sflags != '')}")
    logger.info(f"   Cumulative number of {direction.name} flags set: {np.count_nonzero(dflags == 'w')}")
    logger.info(f"   Cumulative number of {direction.name} convention flags set: {np.count_nonzero(dflags == '1')}")


    return fix_zero_direction # logical_checks

#************************************************************************
def wcc(station: utils.Station, config_dict: dict,
        fix: bool = False, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> np.ndarray:
    """
    Extract the variables and pass to the Wind Cross Checks

    :param Station station: Station Object for the station
    :param str config_dict: string for configuration file (unused here)
    :param bool fix: repair/amend values as a result of logical checks [False]
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: Array of locations where direction data have been corrected
    """

    speed = getattr(station, "wind_speed")
    direction = getattr(station, "wind_direction")

    corrected_locs = logical_checks(speed, direction, fix=fix,
                                    plots=plots, diagnostics=diagnostics)

    return corrected_locs # pcc
