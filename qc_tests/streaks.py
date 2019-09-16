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
import copy
import itertools
import numpy as np


import qc_utils as utils

DATA_COUNT_THRESHOLD = 120


#*********************************************
def plot_streak(times, obs_var, streak_start, streak_end):
    '''
    Plot each streak against surrounding data

    :param array times: datetime array
    :param MetVar obs_var: Meteorological variable object
    :param int streak_start: the location of the streak
    :param int streak_end: the end of the streak

    :returns:
    '''
    import matplotlib.pyplot as plt
        
    pad_start = streak_start - 48
    if pad_start < 0:
        pad_start = 0
    pad_end = streak_end + 48
    if pad_end > len(obs_var.data.compressed()):
        pad_end = len(obs_var.data.compressed())

    # simple plot
    plt.clf()
    plt.plot(times[pad_start: pad_end], obs_var.data.compressed()[pad_start: pad_end], 'ko', )
    plt.plot(times[streak_start: streak_end], obs_var.data.compressed()[streak_start: streak_end], 'ro')    

    plt.ylabel(obs_var.units)
    plt.show()

    return # plot_streak

#************************************************************************
def prepare_data_repeating_string(obs_var, plots=False, diagnostics=False):
    """
    Prepare the data for repeating strings

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # want locations where first differences are zero
    value_diffs = np.ma.diff(obs_var.data.compressed())

    # group the differences
    #     array of (value_diff, count) pairs
    grouped_diffs = np.array([[g[0], len(list(g[1]))] for g in itertools.groupby(value_diffs)])
    
    # all string lengths
    strings, = np.where(grouped_diffs[:, 0] == 0)
    repeated_string_lengths = grouped_diffs[strings, 1] + 1
 
    return repeated_string_lengths, grouped_diffs, strings # prepare_data_repeating_string

#************************************************************************
def get_repeating_string_threshold(obs_var, config_file, plots=False, diagnostics=False):
    """
    Use distribution to determine threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO - how to cope with varying time or measurement resolutions
    # TODO - what if there are only a few "normal" values and otherwise long strings (AGM00060374 20190916)

    this_var = copy.deepcopy(obs_var)
    if obs_var.name == "wind_speed":
        calms, = np.ma.where(this_var.data == 0)
        this_var.data[calms] = utils.MDI
        this_var.data.mask[calms] = True

    # only process further if there is enough data
    if len(this_var.data.compressed()) > 1:

        repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_string(this_var, plots=plots, diagnostics=diagnostics)

        # bin width is 1 as dealing in time index.
        # minimum bin value is 2 as this is the shortest string possible
        threshold = utils.get_critical_values(repeated_string_lengths, binmin=2, binwidth=1.0, plots=plots, diagnostics=diagnostics, title=this_var.name.capitalize(), xlabel="Repeating string length")

        # write out the thresholds...
        utils.write_qc_config(config_file, "STREAK-{}".format(this_var.name), "Straight", "{}".format(threshold), diagnostics=diagnostics)

    else:
        # store high value so threshold never reached
        utils.write_qc_config(config_file, "STREAK-{}".format(this_var.name), "Straight", "{}".format(-utils.MDI), diagnostics=diagnostics)

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
 
    # remove calm periods for wind speeds when (a) calculating thresholds and (b) identifying streaks
    this_var = copy.deepcopy(obs_var)
    if obs_var.name == "wind_speed":
        calms, = np.ma.where(this_var.data == 0)
        this_var.data[calms] = utils.MDI
        this_var.data.mask[calms] = True


    flags = np.array(["" for i in range(this_var.data.shape[0])])
    compressed_flags = np.array(["" for i in range(this_var.data.compressed().shape[0])])

    # retrieve the threshold and store in another dictionary
    threshold = {}
    try:
        th = utils.read_qc_config(config_file, "STREAK-{}".format(this_var.name), "Straight")
        threshold["Straight"] = float(th)
    except KeyError:
        # no threshold set
        print("Threshold missing in config file")
        get_repeating_string_threshold(this_var, config_file, plots=plots, diagnostics=diagnostics)
        th = utils.read_qc_config(config_file, "STREAK-{}".format(this_var.name), "Straight")
        threshold["Straight"] = float(th)

    # only process further if there is enough data
    if len(this_var.data.compressed()) > 1:
        repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_string(this_var, plots=plots, diagnostics=diagnostics)

        # above threshold
        bad, = np.where(repeated_string_lengths >= threshold["Straight"])

        # flag identified strings
        for string in bad:
            start = int(np.sum(grouped_diffs[:strings[string], 1]))
            end = start + int(grouped_diffs[strings[string], 1]) + 1

            compressed_flags[start : end] = "K"

            if plots:
                plot_streak(times, this_var, start, end)

        # undo compression
        flags[this_var.data.mask == False] = compressed_flags
        this_var.flags = utils.insert_flags(this_var.flags, flags)

    if diagnostics:
        
        print("Repeated Strings {}".format(this_var.name))
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
    
        # need to have at least two observations to get a streak
        if len(obs_var.data.compressed()) >= 2:

            # repeating (straight) strings
            if full:
                get_repeating_string_threshold(obs_var, config_file, plots=plots, diagnostics=diagnostics)

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
