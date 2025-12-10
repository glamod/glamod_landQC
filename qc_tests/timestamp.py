"""
Timestamp Check
===============

Checks for instances of more than one reading at the same time, with different values
"""
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils

from qc_tests.spike import generate_differences
#************************************************************************
#*********************************************
def plot_multiple(times: pd.DataFrame,
                  obs_var: utils.MeteorologicalVariable,
                  start: int) -> None:   # pragma: no cover
    '''
    Plot each instance of multiple values against surrounding data

    :param DataFrame times: datetime array
    :param MetVar obs_var: Meteorological variable object
    :param int start: the location of the readings

    :returns:
    '''
    import matplotlib.pyplot as plt

    # simple plot
    plt.clf()
    pad_start = start-24
    if pad_start < 0:
        pad_start = 0
    pad_end = start+1+24
    if pad_end > len(obs_var.data):
        pad_end = len(obs_var.data)

    plt.plot(times[pad_start: pad_end], obs_var.data[pad_start: pad_end], 'k-', marker=".")

    plt.plot(times[start: start+1], obs_var.data[start: start+1], 'r*', ms=10)

    plt.ylabel(obs_var.name.capitalize())
    plt.show()

    # plot_spike


#************************************************************************
def identify_multiple_values(obs_var: utils.MeteorologicalVariable, times: pd.DataFrame,
                             config_dict: dict, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Use config_dict to read in critical values, and then assess to find

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_dict: configuration dictionary to store critical values (UNUSED)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO monthly?

    # use same routine from spike check
    value_diffs, time_diffs, _ = generate_differences(times, obs_var.data)

    # new blank flag array (redone for each difference)
    compressed_flags = np.array(["" for i in range(value_diffs.shape[0])])

    multiple_obs_at_time, = np.where(time_diffs == 0)
    if diagnostics:
        print(f" Number of identical timestamps in {obs_var.name}: {multiple_obs_at_time.shape[0]}")

    if len(multiple_obs_at_time) != 0:
        # to the observations differ for the entries
        suspect_locs, = np.ma.where(value_diffs[multiple_obs_at_time] != 0)

        if len(suspect_locs) > 0:
            # Observations have different values, so not clear which is correct.
            #   Flag both
            # set the first of the obs, then the second which make the diff
            compressed_flags[multiple_obs_at_time[suspect_locs]] = "T"
            compressed_flags[multiple_obs_at_time[suspect_locs]+1] = "T"
        else:
            # Observations have the _same_ value, so add information flag only
            compressed_flags[multiple_obs_at_time] = "2"
            compressed_flags[multiple_obs_at_time+1] = "2"

        # Uncompress the flags & insert
        flags = np.array(["" for i in range(obs_var.data.shape[0])])
        # Offset of 1 from use of the difference arrays
        #   Different to spike offset.  Here want to flag value and following
        #   Spike only want to flag following (for single point spike with large value_diff)
        locs, = np.nonzero(obs_var.data.mask == False)
        flags[locs[:-1]] = compressed_flags
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)
    else:
        flags = np.array(["" for i in range(obs_var.data.shape[0])])

    logger.info(f"Timestamp {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # identify_multiple_values


#************************************************************************
def tsc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Timestamp Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        identify_multiple_values(obs_var, station.times, config_dict, plots=plots, diagnostics=diagnostics)

    return  # tsc

