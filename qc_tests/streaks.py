"""
Repeated Streaks Check
^^^^^^^^^^^^^^^^^^^^^^

   Checks for replication of 
     1. checks for consecutive repeating values
     2. checks if one year has more repeating strings than expected
     3. checks for repeats at a given hour across a number of days
     4. checks for repeats for whole days - all 24 hourly values

   Some thresholds now determined dynamically
"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
import datetime as dt
import itertools


import qc_utils as utils


#************************************************************************
def prepare_data_repeating_string(obs_var, times, plots=False, diagnostics=False):
    """
    Prepare the data for repeating strings

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # want locations where first differences are zero
    value_diffs = np.ma.diff(obs_var.data)

    # group the differences
    #     array of (value_diff, count) pairs
    grouped_diffs = np.array([[g[0], len(list(g[1]))] for g in itertools.groupby(value_diffs)])
    
    # all string lengths
    strings, = np.where(grouped_diffs[:, 0] == 0)
    repeated_string_lengths = grouped_diffs[strings, 1] + 1
 
    return repeated_string_lengths, grouped_diffs, strings # prepare_data_repeating_string

#************************************************************************
def get_repeating_string_threshold(obs_var, times, config_file, plots=False, diagnostics=False):
    """
    Use distribution to determine threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO - how to cope with varying time or measurement resolutions

    repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_string(obs_var, times, plots=plots, diagnostics=diagnostics)

    # bin width is 1 as dealing in time index.
    # minimum bin value is 2 as this is the shortest string possible
    threshold = utils.get_critical_values(repeated_string_lengths, binmin=2, binwidth=1.0, plots=plots, diagnostics=diagnostics)

    # write out the thresholds...
    utils.write_qc_config(config_file, "STREAK-{}".format(obs_var.name), "Straight", "{}".format(threshold), diagnostics=diagnostics)

    return # repeating_string_threshold

#************************************************************************
def repeating_value(obs_var, times, config_file, plots=False, diagnostics=False):
    """
    AKA straight string

    Use config file to read threshold values.  Then find strings which exceed threshold.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
 
    # TODO - remove calm periods for wind speeds when (a) calculating thresholds and (b) identifying streaks

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    # retrieve the threshold and store in another dictionary
    threshold = {}
    try:
        th = utils.read_qc_config(config_file, "STREAK-{}".format(obs_var.name), "Straight")
        threshold["Straight"] = float(th)
    except KeyError:
        # no threshold set
        print("Threshold missing in config file")
        sys.exit(1)

    repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_string(obs_var, times, plots=plots, diagnostics=diagnostics)

    # above threshold
    bad, = np.where(repeated_string_lengths > threshold["Straight"])

    # flag identified strings
    for string in bad:
        start = int(np.sum(grouped_diffs[:strings[string], 1]))
        end = start + int(grouped_diffs[strings[string], 1]) + 1

        flags[start : end] = "K"

    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Repeated Strings {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))


    return # repeating_value


#************************************************************************
def excess_repeating_value():
    
    # more than expected number of short strings during a given period, but each individually not enough to set of main test
    # HadISD - proportion of each year identified by strings, if >5x median, then remove these


    return # excess_repeating_value


#************************************************************************
def day_repeat():

    # any complete repeat of 24hs (as long as not "straight string")


    return # day_repeat


#************************************************************************
def hourly_repeat():

    # repeat of given value at same hour of day for > N days
    # HadISD used fixed threshold.  Perhaps can dynamically calculate?

    return # hourly_repeat



#************************************************************************
def rsc(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Repeating Streak Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # repeating (straight) strings
        if full:
            get_repeating_string_threshold(obs_var, station.times, config_file, plots=plots, diagnostics=diagnostics)

        repeating_value(obs_var, station.times, config_file, plots=plots, diagnostics=diagnostics)

        # more short strings than reasonable
        # excess_repeating_value()

        # repeats at same hour of day
        # hourly_repeat()

        # repeats of whole day
        # day_repeat()

    return # rsc

#************************************************************************
if __name__ == "__main__":
    
    print("checking repeated strings")
#************************************************************************