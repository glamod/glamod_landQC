"""
Timestamp Check
^^^^^^^^^^^

Checks for instances of more than one reading at the same time, with different values
"""
import numpy as np

import qc_utils as utils
#************************************************************************
#*********************************************
def plot_multiple(times, obs_var, start):
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
def identify_multiple_values(obs_var, times, config_file, plots=False, diagnostics=False):
    """
    Use config_file to read in critical values, and then assess to find 

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO check works with missing data (compressed?)
    # TODO monthly?
   
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    time_diffs = np.ma.diff(times)/np.timedelta64(1, "m") # presuming minutes
    value_diffs = np.ma.diff(obs_var.data)

    multiple_obs_at_time, = np.where(time_diffs == 0)
    suspect_locs, = np.ma.where(value_diffs[multiple_obs_at_time] != 0)
                     
    # set the first of the obs, then the second which make the diff
    flags[multiple_obs_at_time[suspect_locs]] == "T"
    flags[multiple_obs_at_time[suspect_locs]+1] == "T"
        
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Timestamp {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # identify_multiple_values


#************************************************************************
def tsc(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Timestamp Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        identify_multiple_values(obs_var, station.times, config_file, plots=plots, diagnostics=diagnostics)

    return  # sc


#************************************************************************
if __name__ == "__main__":
    
    print("checking for more than one value at a single timestamp")
#************************************************************************