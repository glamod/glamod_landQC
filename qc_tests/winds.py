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
    fix_zero_direction, = np.ma.nonzero(np.logical_and(speed.data == 0,
                                                       direction.data.mask == True))
    if fix:
        direction.data[fix_zero_direction] = 0
        direction.data.mask[fix_zero_direction] = False
        logger.info("  Zero direction fixed : {}".format(len(fix_zero_direction)))
    else:
        dflags[fix_zero_direction] = utils.QC_TEST_FLAGS["Wind logical - calm, masked zero direction"]
        logger.info("  Zero direction : {}".format(len(fix_zero_direction)))
        # and set to empty as can be used in parent to copy values to dataframe
        fix_zero_direction = np.array([])

    # negative speeds (can't fix)
    negative_speed = np.ma.nonzero(speed.data < 0)
    sflags[negative_speed] = utils.QC_TEST_FLAGS["Winds"]
    logger.info(f"  Negative speed : {len(negative_speed[0])}")

    # negative directions (don't try to adjust)
    negative_direction = np.ma.nonzero(direction.data < 0)
    dflags[negative_direction] = utils.QC_TEST_FLAGS["Winds"]
    logger.info(f"  Negative direction : {len(negative_direction[0])}")

    # wrapped directions (don't try to adjust)
    wrapped_direction = np.ma.nonzero(direction.data > 360)
    dflags[wrapped_direction] = utils.QC_TEST_FLAGS["Winds"]
    logger.info(f"  Wrapped direction : {len(wrapped_direction[0])}")

    # no direction possible if speed == 0
    bad_direction = np.ma.nonzero(np.logical_and(speed.data == 0,
                                               direction.data != 0))
    dflags[bad_direction] = utils.QC_TEST_FLAGS["Winds"]
    logger.info(f"  Bad direction : {len(bad_direction[0])}")

    # northerlies given as 360, not 0 --> calm
    bad_speed = np.ma.nonzero(np.logical_and(direction.data == 0, speed.data != 0))
    sflags[bad_speed] = utils.QC_TEST_FLAGS["Winds"]
    logger.info(f"  Bad speed : {len(bad_speed[0])}")

    # copy flags into attribute
    speed.store_flags(utils.insert_flags(speed.flags, sflags))
    direction.store_flags(utils.insert_flags(direction.flags, dflags))

    logger.info("Wind Logical")
    logger.info(f"   Cumulative number of {speed.name} flags set: {np.count_nonzero(sflags != '')}")
    logger.info(f"   Cumulative number of {direction.name} flags set: {np.count_nonzero(dflags == 'w')}")
    logger.info(f"   Cumulative number of {direction.name} convention flags set: {np.count_nonzero(dflags == '1')}")

    return fix_zero_direction # logical_checks


def logical_gust(speed: utils.MeteorologicalVariable,
                 gust: utils.MeteorologicalVariable,
                 plots: bool=False,
                 diagnostics: bool=False) -> None:
    """Logical checks on wind gust compared tow ind speed

    Parameters
    ----------
    speed : utils.MeteorologicalVariable
        Wind speed data
    gust : utils.MeteorologicalVariable
        Wind gust data
    plots : bool, optional
        Do plots, by default False
    diagnostics : bool, optional
        Diagnostic output, by default False
    """

    gflags = np.array(["" for i in range(gust.data.shape[0])])
    sflags = np.array(["" for i in range(speed.data.shape[0])])

    # any gusts below zero
    below_zero = np.ma.nonzero(gust.data < 0)
    gflags[below_zero] = "w"
    logger.info(f"  Negative wind gust : {len(below_zero[0])}")

    # any gusts above speed - not clear which is at fault
    low_gust = np.ma.nonzero(speed.data > gust.data)
    gflags[low_gust] = "w"
    sflags[low_gust] = "w"
    logger.info(f"  Wind gust below wind speed: {len(low_gust[0])}")

    gust.store_flags(utils.insert_flags(gust.flags, gflags))
    speed.store_flags(utils.insert_flags(speed.flags, sflags))

    logger.info("Wind Logical - Gust")
    logger.info(f"   Cumulative number of {gust.name} flags set: {np.count_nonzero(gflags != '')}")


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
    gust = getattr(station, "wind_gust")

    corrected_locs = logical_checks(speed, direction, fix=fix,
                                    plots=plots, diagnostics=diagnostics)

    logical_gust(speed, gust, plots=plots, diagnostics=diagnostics)

    return corrected_locs # pcc
