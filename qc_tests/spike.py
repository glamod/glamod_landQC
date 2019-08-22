"""
Spike Check
^^^^^^^^^^^

Checks for short (<=3) observations which are far above/below their immediate neighbours.
"""
import numpy as np
import math
import scipy as sp
import datetime as dt
import copy

import qc_utils as utils
#************************************************************************

MAX_SPIKE_LENGTH = 3

#************************************************************************
def get_critical_values(obs_var, times, config_file, plots=False, diagnostics=False):
    """
    Use distribution to determine critical values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # use all first differences
    # TODO check works with missing data (compressed?)
    # TODO don't run using flagged data
    # TODO monthly?
 
    time_diffs = np.ma.diff(times)/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(obs_var.data)

    # get thresholds for each unique time differences
    unique_diffs = np.unique(time_diffs)

    for t_diff in unique_diffs:

        locs, = np.where(time_diffs == t_diff)

        if len(locs) > utils.MIN_NOBS:
    
            first_differences = value_diffs[locs]

            # fit decay curve to one-sided distribution
            c_value = utils.get_critical_values(first_differences.compressed(), binmin=0, binwidth=0.5, plots=plots, diagnostics=diagnostics, xlabel="First differences", title="Spike - {} - {}h".format(obs_var.name.capitalize(), t_diff))

            # write out the thresholds...
            utils.write_qc_config(config_file, "SPIKE-{}".format(obs_var.name), "{}".format(t_diff), "{}".format(c_value), diagnostics=diagnostics)

    return # get_critical_values

#************************************************************************
def identify_spikes(obs_var, times, config_file, plots=False, diagnostics=False):
    """
    Use config_file to read in critical values, and then assess to find spikes

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO check works with missing data (compressed?)
    # TODO monthly?
    # TODO sort diagnostic output
    # TODO sort plotting output

   
    time_diffs = np.ma.diff(times)/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(obs_var.data)

    # get thresholds for each unique time differences
    unique_diffs = np.unique(time_diffs)

    # retrieve the critical values
    critical_values = {}
    for t_diff in unique_diffs:
        try:
            c_value = utils.read_qc_config(config_file, "SPIKE-{}".format(obs_var.name), "{}".format(t_diff))
            critical_values[t_diff] = float(c_value)
        except KeyError:
            # no critical value for this time difference
            pass

    # pre select for each time difference that can be tested
    for t_diff in unique_diffs:
        # new blank flag array
        flags = np.array(["" for i in range(obs_var.data.shape[0])])
        
        t_locs, = np.where(time_diffs == t_diff)

        try:
            c_locs, = np.where(np.abs(value_diffs[t_locs]) > critical_values[t_diff])
        except:
            # no critical value for this time difference
            continue # to next loop
        
        # potential spikes
        for ps, possible_spike in enumerate(t_locs[c_locs]):
            is_spike = False

            spike_len = 1
            while spike_len <= MAX_SPIKE_LENGTH:
                # test for each possible length to see if identified
                if (possible_spike + spike_len) in c_locs:
                    # check that signs are opposite
                    if np.sign(value_diffs[possible_spike]) != np.sign(value_diffs[possible_spike + spike_len]):
                        is_spike = True
                        break                        
                spike_len += 1

            if is_spike and spike_len > 1:
                # test within spike differences
                within = 1
                while within < spike_len:
                    if value_diffs[possible_spike + within] > (critical_values[t_diff])/2.:
                        is_spike = False 
                    within += 1

            if is_spike:
                # test either side (either one side or the other is too big)
                if value_diffs[possible_spike - 1] > (critical_values[t_diff])/2. or\
                   value_diffs[possible_spike + spike_len + 1] > (critical_values[t_diff])/2.:
                    is_spike = False

            # if the spike is still set, set the flags
            if is_spike:
                flags[possible_spike+1 : possible_spike+1+spike_len] = "S"
                     
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        if diagnostics:

            print("Spike {}".format(obs_var.name))
            print("   Time Difference: {} minutes".format(t_diff))
            print("      Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # identify_spikes


#************************************************************************
def sc(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the World Record Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    for var in var_list:

        obs_var = getattr(station, var)

        # decide whether to recalculate
        if full:
            get_critical_values(obs_var, station.times, config_file, plots=plots, diagnostics=diagnostics)

        identify_spikes(obs_var, station.times, config_file, plots=plots, diagnostics=diagnostics)

    return  # sc


#************************************************************************
if __name__ == "__main__":
    
    print("checking for short period spikes")
#************************************************************************
