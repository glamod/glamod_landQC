"""
Timestamp Check
^^^^^^^^^^^^^^^

Checks for instances of more than one reading at the same time, with different values
"""
import numpy as np

import qc_utils as utils
from spike import generate_differences
#************************************************************************
#*********************************************
def plot_multiple(times: np.array, obs_var: utils.Meteorological_Variable, start: int) -> None:
    '''
    Plot each instance of multiple values against surrounding data

    :param array times: datetime array
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

    return # plot_spike

#************************************************************************
def identify_multiple_values(obs_var: utils.Meteorological_Variable, times: np.array,
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
#    if diagnostics:
#        print("number of identical timestamps {}".format(multiple_obs_at_time.shape[0]))

    suspect_locs, = np.ma.where(value_diffs[multiple_obs_at_time] != 0)

    # set the first of the obs, then the second which make the diff
    compressed_flags[multiple_obs_at_time[suspect_locs]] = "T"
    compressed_flags[multiple_obs_at_time[suspect_locs]+1] = "T"

    # Uncompress the flags & insert
    flags = np.array(["" for i in range(obs_var.data.shape[0])])
    # Offset of 1 from use of the difference arrays
    #   Different to spike offset.  Here want to flag value and following
    #   Spike only want to flag following (for single point spike with large value_diff)
    locs, = np.nonzero(obs_var.data.mask == False)
    flags[locs[:-1]] = compressed_flags
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:

        print("Timestamp {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # identify_multiple_values


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


#************************************************************************
if __name__ == "__main__":

    print("checking for more than one value at a single timestamp")
#************************************************************************
