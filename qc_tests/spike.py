"""
Spike Check
===========

Checks for short (<=3) observations which are far above/below their immediate neighbours.
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************

MAX_SPIKE_LENGTH = 3

# Up to
TIME_DIFF_RANGES = np.array([[1, 5],  # 4
                             [6, 10],  # 5
                             [11, 15],  # 5
                             [16, 20],  # 10
                             [21, 30],  # 10
                             [31, 45],  # 15
                             [46, 60],  # 15
                             [61, 90],  # 30
                             [91, 120],  # 30  [total of 2h so far]
                             [121, 180],  # 60 (1h)
                             [181, 300],  # 120 (2h)
                             [301, 540],  # 240 (4h)
                             [541, 900],  # 360 (6h) [total of 15h so far]
                             [901, 1440],  # 540 (9h) [total of 1d so far]
                             [1441, 2160],  # 12h
                             [2161, 2880],  # 12h [total of 2d so far]
                             [2880, 4320],  # 24h [total of 3d]
                             [4320, 7200],  # 48h [total of 5d]
                             ])


#*********************************************
def plot_spike(times: np.ndarray, obs_var: utils.Meteorological_Variable,
               spike_start: int, spike_length: int) -> None:
    '''
    Plot each spike against surrounding data

    :param array times: datetime array
    :param MetVar obs_var: Meteorological variable object
    :param int spike_start: the location of the spike
    :param int spike_length: the length of the spike

    :returns:
    '''
    import matplotlib.pyplot as plt

    # Need to apply masks to match counting in identification steps
    mdata = obs_var.data.compressed()
    mtimes = np.ma.masked_array(times, mask=obs_var.data.mask).compressed()

    # simple plot
    plt.clf()
    pad_start = spike_start-24
    if pad_start < 0:
        pad_start = 0
    pad_end = spike_start+spike_length+24
    if pad_end > len(mdata):
        pad_end = len(mdata)

    plt.plot(mtimes[pad_start: pad_end],
             mdata[pad_start: pad_end], 'k-', marker=".")

    plt.plot(mtimes[spike_start: spike_start+spike_length],
             mdata[spike_start: spike_start+spike_length], 'r*', ms=10)

    plt.ylabel(obs_var.name.capitalize())
    plt.show()

    return # plot_spike

#************************************************************************
def calculate_critical_values(obs_var: utils.Meteorological_Variable, times: np.ndarray,
                              config_dict: dict, plots: bool=False, diagnostics: bool=False) -> None:
    """
    Use distribution to determine critical values.  Then also store in config dictionary.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    # generate the first difference metrics
    value_diffs, time_diffs, unique_diffs = generate_differences(times, obs_var.data)

    if 0 in unique_diffs:
        # Not a spike or jump, but 2 values at the same time.
        #  should be zero value difference, so fitting histogram not going to work
        #  handled in separate test
        logger.warning("Spike Check - zero time difference between two timestamps. Check")
        return


    # Now go through each unique time difference to calculate critical values
    for (lower, upper) in TIME_DIFF_RANGES:

        locs, = np.where(np.logical_and(time_diffs >= lower,
                                        time_diffs <= upper))

        first_differences = value_diffs[locs]

        # ensure sufficient non-masked observations
        if len(first_differences.compressed()) >= utils.DATA_COUNT_THRESHOLD:

            # fit decay curve to one-sided distribution
            c_value = utils.get_critical_values(first_differences.compressed(), binmin=0, binwidth=0.5,
                                                plots=plots, diagnostics=diagnostics,
                                                xlabel="First differences",
                                                title=f"Spike - {obs_var.name.capitalize()}: {lower}-{upper}min")

            # write out the thresholds...
            try:
                config_dict[f"SPIKE-{obs_var.name}"][f"{lower}-{upper}"] = float(c_value)
            except KeyError:
                CD_diff = {f"{lower}-{upper}" : float(c_value)}
                config_dict[f"SPIKE-{obs_var.name}"] = CD_diff

            logger.debug(f"   Time Difference: {lower}-{upper} minutes")
            logger.debug(f"      Number of obs: {len(first_differences.compressed())}, threshold: {c_value}")

        else:
            logger.debug(f"   Time Difference: {lower}-{upper} minutes")
            logger.debug(f"      Number of obs insufficient: {len(first_differences.compressed())} < {utils.DATA_COUNT_THRESHOLD}")


#************************************************************************
def retreive_critical_values(config_dict: dict, name: str) -> dict:
    """
    Read the config dictionary to pull out the critical values

    :param dict config_dict: the dictionary (hopefully) holding the values
    :param str name: the variable in question

    :returns: dict
    """
    # Read in the dictionary
    critical_values = {}

    for (lower, upper) in TIME_DIFF_RANGES:
        try:
            c_value = config_dict[f"SPIKE-{name}"][f"{lower}-{upper}"]
            critical_values[f"{lower}-{upper}"] = float(c_value)
        except KeyError:
            # no critical value for this time difference
            pass

    return critical_values


#************************************************************************
def assess_potential_spike(time_diffs: np.ndarray, value_diffs: np.ndarray,
                           possible_in_spike: int, critical_values: dict) -> tuple[bool, int]:
    """
    Check if the jump up is the beginning of a spike (so, look for a large enough jump down within
    the permitted time frame).

    :param array time_diffs: time differences to look at
    :param array value_diffs: value first differences
    :param int possible_in_spike: location of potential start of spike
    :param dict critical_values: threshold values for this spike

    :returns: (bool, int) of spike and length
    """

    is_spike = False
    spike_len = 1

    while spike_len <= MAX_SPIKE_LENGTH:
        # test for each possible length to see if identified
        try:
            out_spike_t_diff = time_diffs[possible_in_spike + spike_len]
            possible_out_spike = value_diffs[possible_in_spike + spike_len]
        except IndexError:
            # TODO got to end of data run, can't test final value at the moment
            break

        try:
            # find critical value for time-difference of way out of spike
            out_critical_value = critical_values[out_spike_t_diff]
        except KeyError:
            # don't have a value for this time difference, so use the maximum of all as a proxy
            out_critical_value = max(critical_values.values())
        else:
            # time or value difference masked
            out_critical_value = max(critical_values.values())

        if np.abs(possible_out_spike) > out_critical_value:
            # check that the signs are opposite
            if np.sign(value_diffs[possible_in_spike]) != np.sign(value_diffs[possible_in_spike + spike_len]):
                is_spike = True
                break

        spike_len += 1

    return is_spike, spike_len


#************************************************************************
def assess_inside_spike(time_diffs: np.ndarray, value_diffs: np.ndarray,
                        possible_in_spike: int, critical_values: dict,
                        is_spike: bool, spike_len: int) -> bool:
    """
    Check if points inside the spike don't vary too much (low noise)

    :param array time_diffs: time differences to look at
    :param array value_diffs: value first differences
    :param int possible_in_spike: location of potential start of spike
    :param dict critical_values: threshold values for this spike
    :param bool is_spike: flag to indicate presence of spike
    :param int spike_len: length of spike (number of time stamps)

    :returns: bool
    """

    # test within spike differences (chosing correct time difference)
    within = 1
    # if single timestep spike, then this assessment not necessary
    while within < spike_len:
        # find the time difference between appropriate neighbouring points
        within_t_diff = time_diffs[possible_in_spike + within]

        # obtain relevant critical value for time diff
        try:
            within_critical_value = critical_values[within_t_diff]
        except KeyError:
            # don't have a value for this time difference, so use the maximum of all as a proxy
            within_critical_value = max(critical_values.values())
        else:
            # time difference masked
            within_critical_value = max(critical_values.values())

        # check if any differences are greater than 1/2 critical value for time diff

        if np.abs(value_diffs[possible_in_spike + within]) > within_critical_value/2.:
            is_spike = False

        within += 1

    return is_spike


#************************************************************************
def assess_outside_spike(time_diffs: np.ndarray, value_diffs: np.ndarray,
                        possible_in_spike: int, critical_values: dict,
                        is_spike: bool, spike_len: int) -> tuple[bool, int]:
    """
    Check if points outside the spike don't vary too much (low noise).
    Using "side" to act as parameter for the timestamps before/after the spike

    :param array time_diffs: time differences to look at
    :param array value_diffs: value first differences
    :param int possible_in_spike: location of potential start of spike
    :param dict critical_values: threshold values for this spike

    :returns: (bool, int) of spike and length
    """

    # test either side (either before or after is too big)

    for outside in [-1, spike_len + 1]:
        # if outside = -1, then before
        # if outside = length+1, then after

        try:
            outside_t_diff = time_diffs[possible_in_spike + outside]
            outside_critical_value = critical_values[outside_t_diff]
        except KeyError:
            # don't have a value for this time difference, so use the maximum of all as a proxy
            outside_critical_value = max(critical_values.values())
        except IndexError:
            # off the front/back of the data array
            outside_critical_value = max(critical_values.values())

        try:
            if np.abs(value_diffs[possible_in_spike + outside]) > outside_critical_value/2.:
                # spike fails test
                is_spike = False

        except IndexError:
            # off the front/back of the data array
            pass

    return is_spike


#************************************************************************
def generate_differences(times: np.ndarray,
                         data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the first differences for the times and the data values

    :param array times: array of times (usually in minutes)
    :param array data: array of data

    :returns: (value_diffs, time_diffs, unique_diffs)
    """
    # apply mask to times too
    masked_times = np.ma.masked_array(times, mask=data.mask)

    # Using .compressed() to remove any masked points
    #   If masks remain, then one of the differences will be masked, and hence unassessed
    time_diffs = np.ma.diff(masked_times.compressed())/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(data.compressed())

    if len(value_diffs.mask.shape) == 0:
        # single mask value, replace with array of True/False's
        if value_diffs.mask:
            value_diffs.mask = np.ones(value_diffs.shape)
        else:
            value_diffs.mask = np.zeros(value_diffs.shape)

    # get thresholds for each unique time differences
    unique_diffs = np.unique(time_diffs.compressed())

    return value_diffs, time_diffs, unique_diffs


#************************************************************************
def identify_spikes(obs_var: utils.Meteorological_Variable, times: np.ndarray, config_dict: dict,
                    plots: bool = False, diagnostics: bool = False) -> None:
    """
    Use config_dict to read in critical values, and then assess to find spikes

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # get the first differences
    value_diffs, time_diffs, unique_diffs = generate_differences(times, obs_var.data)

    # retrieve the critical values
    critical_values = retreive_critical_values(config_dict, obs_var.name)

    # if none have been read, give an option to calculate in case that was the reason for none
    if len(critical_values) == 0:
        calculate_critical_values(obs_var, times, config_dict, plots=plots, diagnostics=diagnostics)
        critical_values = retreive_critical_values(config_dict, obs_var.name)

    # pre select for each time difference that can be testedlen
    for (lower, upper) in TIME_DIFF_RANGES:

        # new blank flag array (redone for each difference)
        compressed_flags = np.array(["" for i in range(value_diffs.shape[0])])

        # select the locations above critical value and with the relevant time difference
        try:
            potential_spike_locs, = np.nonzero(
                np.logical_and((np.abs(value_diffs) > critical_values[f"{lower}-{upper}"]),
                                np.logical_and((time_diffs >= lower),
                                               (time_diffs <= upper))
                                )
                )
        except:
            # no critical value for this time difference
            continue # to next loop

        # TODO - sort spikes at very beginning or very end of sequence,
        #    when don't have a departure from/return to a normal level

        # assess identified potential spikes
        for possible_in_spike in potential_spike_locs:

            is_spike, spike_len = assess_potential_spike(time_diffs, value_diffs,
                                                         possible_in_spike, critical_values)

            if is_spike and spike_len >= 1:
                is_spike = assess_inside_spike(time_diffs, value_diffs,
                                               possible_in_spike, critical_values, is_spike, spike_len)

            if is_spike:
                is_spike = assess_outside_spike(time_diffs, value_diffs,
                                               possible_in_spike, critical_values, is_spike, spike_len)

            # if the spike is still set, set the flags
            if is_spike:
                compressed_flags[possible_in_spike : possible_in_spike+spike_len] = "S"

                # diagnostic plots
                if plots:
                    plot_spike(times, obs_var, possible_in_spike+1, spike_len)

        # Uncompress the flags & insert
        flags = np.array(["" for i in range(obs_var.data.shape[0])])
        # offset of 1 from use of the difference arrays
        locs, = np.nonzero(obs_var.data.mask == False)
        flags[locs[1:]] = compressed_flags
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        logger.info(f"Spike {obs_var.name}")
        logger.info(f"   Time Difference: {lower}-{upper} minutes")
        logger.info(f"      Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # identify_spikes


#************************************************************************
def sc(station: utils.Meteorological_Variable, var_list: list, config_dict: dict,
       full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Spike Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    for var in var_list:

        obs_var = getattr(station, var)

        # decide whether to recalculate
        if full:
            calculate_critical_values(obs_var, station.times, config_dict, plots=plots, diagnostics=diagnostics)

        identify_spikes(obs_var, station.times, config_dict, plots=plots, diagnostics=diagnostics)

    return  # sc

