"""
Humidity Cross Checks
^^^^^^^^^^^^^^^^^^^^^

1. Check and flag instances of super saturation
2. Check and flag instances of dew point depression
"""
#************************************************************************
import itertools
import numpy as np

import qc_utils as utils


#************************************************************************
def prepare_data_repeating_dpd(locs, plots=False, diagnostics=False):
    """
    Prepare the data for repeating strings

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # want locations where first differences are zero
    index_diffs = np.ma.diff(locs)

    # group the differences
    #     array of (value_diff, count) pairs
    grouped_diffs = np.array([[g[0], len(list(g[1]))] for g in itertools.groupby(index_diffs)])

    # all adjacent values, hence difference in array-index is 1
    strings, = np.where(grouped_diffs[:, 0] == 1)
    repeated_string_lengths = grouped_diffs[strings, 1] + 1
 
    return repeated_string_lengths, grouped_diffs, strings # prepare_data_repeating_dpd

#************************************************************************
def get_repeating_dpd_threshold(temperatures, dewpoints, config_file, plots=False, diagnostics=False):
    """
    Use distribution to determine threshold values.  Then also store in config file.

    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    dpd = temperatures.data - dewpoints.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.where(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_dpd(locs, plots=plots, diagnostics=diagnostics)

        # bin width is 1 as dealing with the index.
        # minimum bin value is 2 as this is the shortest string possible
        threshold = utils.get_critical_values(repeated_string_lengths, binmin=2, binwidth=1.0, plots=plots, diagnostics=diagnostics, title="DPD streak length", xlabel="Repeating DPD length")

        # write out the thresholds...
        utils.write_qc_config(config_file, "HUMIDITY", "DPD", "{}".format(threshold), diagnostics=diagnostics)
    else:
        # store high value so threshold never reached
        utils.write_qc_config(config_file, "HUMIDITY", "DPD", "{}".format(-utils.MDI), diagnostics=diagnostics)


    return # repeating_dpd_threshold

#*********************************************
def plot_humidities(T, D, times, bad):
    '''
    Plot each observation of SSS or DPD against surrounding data

    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints
    :param array times: datetime array
    :param int bad: the location of SSS

    :returns:
    '''
    import matplotlib.pyplot as plt

    pad_start = bad - 24
    if pad_start < 0:
        pad_start = 0
    pad_end = bad + 24
    if pad_end > len(T.data):
        pad_end = len(T.data)

    # simple plot
    plt.clf()
    plt.plot(times[pad_start : pad_end], T.data[pad_start : pad_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start : pad_end], D.data[pad_start : pad_end], 'b-', marker=".", label=D.name.capitalize())
    plt.plot(times[bad], D.data[bad], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(T.units)
    plt.show()

    return # plot_humidities

#*********************************************
def plot_humidity_streak(times, T, D, streak_start, streak_end):
    '''
    Plot each streak against surrounding data

    :param array times: datetime array
    :param MetVar T: Meteorological variable object - temperatures
    :param MetVar D: Meteorological variable object - dewpoints
    :param int streak_start: the location of the streak
    :param int streak_end: the end of the DPD streak

    :returns:
    '''
    import matplotlib.pyplot as plt

    pad_start = streak_start - 48
    if pad_start < 0:
        pad_start = 0
    pad_end = streak_end + 48
    if pad_end > len(T.data.compressed()):
        pad_end = len(T.data.compressed())

    # simple plot
    plt.clf()
    plt.plot(times[pad_start: pad_end], T.data.compressed()[pad_start: pad_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[pad_start: pad_end], D.data.compressed()[pad_start: pad_end], 'b-', marker=".", label=D.name.capitalize())
    plt.plot(times[streak_start: streak_end], T_var.data.compressed()[streak_start: streak_end], 'k-', marker=".", label=T.name.capitalize())
    plt.plot(times[streak_start: streak_end], D_var.data.compressed()[streak_start: streak_end], 'b-', marker=".", label=D.name.capitalize())

    plt.ylabel(obs_var.units)
    plt.show()

    return # plot_humidity_streak

#************************************************************************
def super_saturation_check(times, temperatures, dewpoints, plots=False, diagnostics=False):
    """
    Flag locations where dewpoint is greater than air temperature

    :param array times: datetime array
    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    sss, = np.ma.where(dewpoints.data > temperatures.data)

    flags[sss] = "h"

    # only flag the dewpoints
    dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)

    # TODO - what if enough of a month is flagged...

    # diagnostic plots
    if plots:
        for bad in sss:
            plot_humidities(temperatures, dewpoints, times, bad)

    if diagnostics:

        print("Supersaturation {}".format(dewpoints.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # super_saturation_check

#************************************************************************
def dew_point_depression_streak(times, temperatures, dewpoints, config_file, plots=False, diagnostics=False):
    """
    Flag locations where dewpoint equals air temperature

    :param array times: datetime array
    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    # retrieve the threshold and store in another dictionary
    try:
        th = utils.read_qc_config(config_file, "HUMIDITY", "DPD")
        threshold = float(th)
    except KeyError:
        # no threshold set
        print("Threshold missing in config file")
        get_repeating_dpd_threshold(temperatures, dewpoints, config_file, plots=plots, diagnostics=diagnostics)
        th = utils.read_qc_config(config_file, "HUMIDITY", "DPD")
        threshold = float(th)


    dpd = temperatures.data - dewpoints.data

    # find only the DPD=0 locations, and then see if there are streaks
    locs, = np.ma.where(dpd == 0)

    # only process further if there are enough locations
    if len(locs) > 1:
        repeated_string_lengths, grouped_diffs, strings = prepare_data_repeating_dpd(locs, plots=plots, diagnostics=diagnostics)

        # above threshold
        bad, = np.where(repeated_string_lengths >= threshold)

        # flag identified strings
        for string in bad:
            start = int(np.sum(grouped_diffs[:strings[string], 1]))
            end = start + int(grouped_diffs[strings[string], 1]) + 1

            flags[start : end] = "h"

            if plots:
                plot_humiditystreak(times, temperatures, dewpoints, start, end)

        # only flag the dewpoints
        dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)

    if diagnostics:

        print("Dewpoint Depression {}".format(dewpoints.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # dew_point_depression_streak

#************************************************************************
def hcc(station, config_file, full=False, plots=False, diagnostics=False):
    """
    Extract the variables and pass to the Humidity Cross Checks

    :param Station station: Station Object for the station
    :param str configfile: string for configuration file
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")

    # Super Saturation
#    super_saturation_check(station.times, temperatures, dewpoints, plots=plots, diagnostics=diagnostics)

    # Dew Point Depression
    #    Note, won't have cloud-base or past-significant-weather
    #    Note, currently don't have precipitation information
    if full:
        get_repeating_dpd_threshold(temperatures, dewpoints, config_file, plots=plots, diagnostics=diagnostics)
    dew_point_depression_streak(station.times, temperatures, dewpoints, config_file, plots=plots, diagnostics=diagnostics)

    # dew point cut-offs (HadISD) not run
    #  greater chance of removing good observations 
    #  18 July 2019 RJHD

    return # hcc

#************************************************************************
if __name__ == "__main__":

    print("humidity cross checks")
#************************************************************************
