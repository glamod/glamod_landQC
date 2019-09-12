"""
Spike Check
^^^^^^^^^^^

Checks for short (<=3) observations which are far above/below their immediate neighbours.
"""
import numpy as np

import qc_utils as utils
#************************************************************************

MAX_SPIKE_LENGTH = 3

#*********************************************
def plot_spike(times, obs_var, spike_start, spike_length):
    '''
    Plot each spike against surrounding data

    :param array times: datetime array
    :param MetVar obs_var: Meteorological variable object
    :param int spike_start: the location of the spike
    :param int spike_length: the length of the spike

    :returns:
    '''
    import matplotlib.pyplot as plt
        
    # simple plot
    plt.clf()
    pad_start = spike_start-24
    if pad_start < 0:
        pad_start = 0
    pad_end = spike_start+spike_length+24
    if pad_end > len(obs_var.data):
        pad_end = len(obs_var.data)

    plt.plot(times[pad_start: pad_end], obs_var.data[pad_start: pad_end], 'k-', marker=".")        

    plt.plot(times[spike_start: spike_start+spike_length], obs_var.data[spike_start: spike_start+spike_length], 'r*', ms=10)    

    plt.ylabel(obs_var.name.capitalize())
    plt.show()

    return # plot_spike

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
 
    masked_times = np.ma.masked_array(times, mask = obs_var.data.mask)

    time_diffs = np.ma.diff(masked_times)/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(obs_var.data)

    # get thresholds for each unique time differences
    unique_diffs = np.unique(time_diffs.compressed())

    for t_diff in unique_diffs:

        if t_diff == 0:
            # not a spike or jump, but 2 values at the same time.
            #  should be zero value difference, so fitting histogram not going to work
            #  handled in separate test
            print("test")
            continue

        locs, = np.where(time_diffs == t_diff)

        first_differences = value_diffs[locs]

        # ensure sufficient non-masked observations
        if len(first_differences.compressed()) > utils.MIN_NOBS:

            # fit decay curve to one-sided distribution
            c_value = utils.get_critical_values(first_differences.compressed(), binmin=0, binwidth=0.5, plots=plots, diagnostics=diagnostics, xlabel="First differences", title="Spike - {} - {}m".format(obs_var.name.capitalize(), t_diff))

            # write out the thresholds...
            utils.write_qc_config(config_file, "SPIKE-{}".format(obs_var.name), "{}".format(t_diff), "{}".format(c_value), diagnostics=diagnostics)
            if diagnostics:
                print("   Time Difference: {} minutes".format(t_diff))
                print("      Number of obs: {}, threshold: {}".format(len(first_differences.compressed()), c_value))
        else:
            if diagnostics:
                print("   Time Difference: {} minutes".format(t_diff))
                print("      Number of obs insufficient: {} < {}".format(len(first_differences.compressed()), utils.MIN_NOBS))


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

    masked_times = np.ma.masked_array(times, mask = obs_var.data.mask)
   
    time_diffs = np.ma.diff(masked_times)/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(obs_var.data)

    # get thresholds for each unique time differences
    unique_diffs = np.unique(time_diffs.compressed())

    # retrieve the critical values
    critical_values = {}
    for t_diff in unique_diffs:
        try:
            c_value = utils.read_qc_config(config_file, "SPIKE-{}".format(obs_var.name), "{}".format(t_diff))
            critical_values[t_diff] = float(c_value)
        except KeyError:
            # no critical value for this time difference
            pass

    # if none have been read, give an option to calculate in case that was the reason for none
    if len(critical_values) == 0:
        get_critical_values(obs_var, times, config_file, plots=plots, diagnostics=diagnostics)

        # and try again
        for t_diff in unique_diffs:
            try:
                c_value = utils.read_qc_config(config_file, "SPIKE-{}".format(obs_var.name), "{}".format(t_diff))
                critical_values[t_diff] = float(c_value)
            except KeyError:
                # no critical value for this time difference
                pass


    # pre select for each time difference that can be tested
    for t_diff in unique_diffs:
        if t_diff == 0:
            # not a spike or jump, but 2 values at the same time.
            #  should be zero value difference, so fitting histogram not going to work
            #  handled in separate test
            continue

        # new blank flag array
        flags = np.array(["" for i in range(obs_var.data.shape[0])])
        
        t_locs, = np.where(time_diffs == t_diff)

        try:
            c_locs, = np.where(np.abs(value_diffs[t_locs]) > critical_values[t_diff])
        except:
            # no critical value for this time difference
            continue # to next loop
        
        # potential spikes
        for ps, possible_in_spike in enumerate(t_locs[c_locs]):
            is_spike = False

            spike_len = 1
            while spike_len <= MAX_SPIKE_LENGTH:
                # test for each possible length to see if identified
                out_spike_t_diff = time_diffs[possible_in_spike + spike_len]
                possible_out_spike = value_diffs[possible_in_spike + spike_len]
                try:
                    # find critical value for time-difference of way out of spike
                    out_critical_value = critical_values[out_spike_t_diff]
                except KeyError:
                    # don't have a value for this time difference, so use the maximum of all as a proxy
                    out_critical_value = max(critical_values.values())
                    
                if np.abs(possible_out_spike) > out_critical_value:
                    # check that the signs are opposite
                    if np.sign(value_diffs[possible_in_spike]) != np.sign(value_diffs[possible_in_spike + spike_len]):
                        is_spike = True
                        break

                spike_len += 1

            if is_spike and spike_len >= 1:
                # test within spike differences (chosing correct time difference)
                within = 1
                while within < spike_len:
                    within_t_diff = time_diffs[possible_in_spike + within]
                    try:
                        within_critical_value = critical_values[within_t_diff]
                        if value_diffs[possible_in_spike + within] > within_critical_value/2.:
                            is_spike = False 
                    except KeyError:
                        # don't have a value for this time difference, so use the maximum of all as a proxy
                        within_critical_value = max(critical_values.values())

                    if value_diffs[possible_in_spike + within] > within_critical_value/2.:
                        is_spike = False 
                        
                    within += 1

            if is_spike:
                # test either side (either before or after is too big)
                before_t_diff = time_diffs[possible_in_spike - 1]
                try:
                    before_critical_value = critical_values[before_t_diff]
                except KeyError:
                    # don't have a value for this time difference, so use the maximum of all as a proxy
                    before_critical_value = max(critical_values.values())                    

                after_t_diff = time_diffs[possible_in_spike + spike_len + 1]
                try:
                    after_critical_value = critical_values[after_t_diff]
                except KeyError:
                    # don't have a value for this time difference, so use the maximum of all as a proxy
                    after_critical_value = max(critical_values.values())

                if value_diffs[possible_in_spike - 1] > before_critical_value/2. or\
                   value_diffs[possible_in_spike + spike_len + 1] > after_critical_value/2.:
                    is_spike = False

            # if the spike is still set, set the flags
            if is_spike:
                # "+1" because of difference arrays
                flags[possible_in_spike+1 : possible_in_spike+1+spike_len] = "S"

                # diagnostic plots
                if plots:
                    plot_spike(times, obs_var, possible_in_spike+1, spike_len)                     

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        if diagnostics:

            print("Spike {}".format(obs_var.name))
            print("   Time Difference: {} minutes".format(t_diff))
            print("      Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # identify_spikes


#************************************************************************
def sc(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Spike Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    for var in ["wind_speed"]:# var_list:

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
