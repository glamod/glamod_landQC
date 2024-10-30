"""
High Flag Rate Check
^^^^^^^^^^^^^^^^^^^^

Check for high flagging rates in each obs variable.  Indicate to 
withhold station if in more than one.

This is for the whole timeseries of data, whereas clean_up.py assesses monthly

For the pressure and wind synergistic variables, if one has these flags set,
then the other will be set too.

Run at the end of both the internal and neighbour checks.
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils

#************************************************************************
def set_synergistic_flags(station: utils.Station, var: str) -> None:
    """
    Set the flags on a synergistic variable. High flagging rate set
    on all obs in the variable, so no need to do any extra comparison

    :param Station station: Station Object for the station
    :param str var: name of variable
    """
    obs_var = getattr(station, var)

    new_flags = np.array(["" for i in range(obs_var.data.shape[0])])
    # old_flags = obs_var.flags
    obs_locs, = np.where(obs_var.data.mask == False)

    if obs_locs.shape[0] > 10 * utils.DATA_COUNT_THRESHOLD:
        # require sufficient observations to make a flagged fraction useful.

        # As synergistically flagged, add to all flags.
        new_flags[obs_locs] = "H"

    obs_var.flags = utils.insert_flags(obs_var.flags, new_flags)

    return # set_synergistic_flags

#************************************************************************
def high_flag_rate(obs_var: utils.Meteorological_Variable,
                   plots: bool = False, diagnostics: bool = False) -> tuple[np.ndarray, bool]:
    """
    Check for high flag rates, and remove any remaining observations.

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: (array, bool) - new flags and if any have been set
    """
    any_flags_set = False
    new_flags = np.array(["" for i in range(obs_var.data.shape[0])])
    old_flags = obs_var.flags
    obs_locs, = np.where(obs_var.data.mask == False)

    if obs_locs.shape[0] > 10 * utils.DATA_COUNT_THRESHOLD:
        # require sufficient observations to make a flagged fraction useful.

        # If already flagged on internal run, return with dummy results.
        flag_set = np.unique(old_flags[obs_locs]) # Flags per obs.
        unique_flags = set("".join(flag_set)) # Unique set of flag letters.
        if "H" in unique_flags:
            # This test has been run before on this variable, so don't do again.
            return new_flags, any_flags_set

        flagged, = np.where(old_flags[obs_locs] != "")

        if flagged.shape[0] / obs_locs.shape[0] > utils.HIGH_FLAGGING:
            if diagnostics:
                print(f" {obs_var.name} flagging rate of {100*(flagged.shape[0] / obs_locs.shape[0]):5.1f}%")
            # Set flags only obs currently unflagged.
            unflagged, = np.where(old_flags[obs_locs] == "")
            new_flags[obs_locs[unflagged]] = "H"
            any_flags_set = True

    logger.info(f"High Flag Rate {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(new_flags == 'H')[0])}")

    return new_flags, any_flags_set # high_flag_rate

#************************************************************************
def hfr(station: utils.Station, var_list: list, full: bool = False, plots: bool = False, diagnostics: bool = False) -> int:
    """
    Run through the variables and pass to the High Flag Rate Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: int : number of variables on which these flags have been set
    """
    vars_set = [] # Keep track of where these flags are set.

    for var in var_list:

        obs_var = getattr(station, var)

        flags, any_set = high_flag_rate(obs_var, plots=plots, diagnostics=diagnostics)

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        if any_set:
            vars_set += [var]

    # Now double check the list of variables where "H" flags have been set.
    #  If one of a synergistic pair is, then do the other (wind speed/direction,
    #  sea/station level pressure).
    # Using exclusive or.  This only passes if one is True and the other is False.
    if ("sea_level_pressure" in vars_set) is not ("station_level_pressure" in vars_set):

        if "sea_level_pressure" in vars_set:
            set_synergistic_flags(station, "station_level_pressure")
        elif "station_level_pressure" in vars_set:
            set_synergistic_flags(station, "sea_level_pressure")

    if ("wind_speed" in vars_set) is not ("wind_direction" in vars_set):

        if "wind_speed" in vars_set:
            set_synergistic_flags(station, "wind_direction")
        elif "wind_direction" in vars_set:
            set_synergistic_flags(station, "wind_speed")

    # For synergistically flagged, just count once, so this return is correct.
    return len(vars_set) # hfr

#************************************************************************
if __name__ == "__main__":

    print("checking for high flagging rates")
#************************************************************************
