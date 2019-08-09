"""
Odd Cluster Checks
^^^^^^^^^^^^^^^^^^

Find isolated data (single points and small runs)
"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
import datetime as dt

import qc_utils as utils
#************************************************************************

MAX_LENGTH_OBS = 6 # up to X data points
MAX_LENGTH_TIME = 24 # within a max of Y hours
MIN_SEPARATION = 48 # separated by Z hours on either side from other data


#************************************************************************
def flag_clusters(obs_var, station, plots=False, diagnostics=False):

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    time_differences = np.diff(station.times)/np.timedelta64(1, "m")

    potential_cluster_ends, = np.where(time_differences >= MIN_SEPARATION * 60)

    # TODO - need explicit checks for start and end of timeseries
    for ce, cluster_end in enumerate(potential_cluster_ends):

        if ce == 0:
            # check if cluster at start of series (long gap after a first few points)
            cluster_length = station.times.iloc[cluster_end]-station.times.iloc[0] 
            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[:cluster_end+1]) < MAX_LENGTH_OBS:
                    flags[:cluster_end+1] = "o"

        elif ce == len(potential_cluster_ends) - 1:

            # check if cluster at end of series (long gap before last few points)
            cluster_length = station.times.iloc[-1] - station.times.iloc[cluster_end+1] # add one to find cluster start!
            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[cluster_end+1:]) < MAX_LENGTH_OBS:
                    flags[cluster_end+1:] = "o"

        if ce > 0:
            # check for cluster in series.
            #  use previous gap > MIN_SEPARATION to define cluster and check length
            cluster_length = station.times.iloc[cluster_end] - station.times.iloc[potential_cluster_ends[ce-1]+1] # add one to find cluster start!
            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[potential_cluster_ends[ce-1]+1: cluster_end+1]) < MAX_LENGTH_OBS:
                    flags[potential_cluster_ends[ce-1]+1: cluster_end+1] = "o"

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Odd Cluster {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # flag_clusters

#************************************************************************
def occ(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Odd Cluster Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file (unused at the moment)
    :param bool full: run a full update (unused at the moment)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        flag_clusters(obs_var, station, plots=plots, diagnostics=diagnostics)


    return # dgc

#************************************************************************
if __name__ == "__main__":
    
    print("checking gaps in distributions")
#************************************************************************
