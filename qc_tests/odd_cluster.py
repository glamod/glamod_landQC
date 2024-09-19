"""
Odd Cluster Checks
^^^^^^^^^^^^^^^^^^

Find isolated data (single points and small runs)
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************

MAX_LENGTH_OBS = 6 # up to X data points
MAX_LENGTH_TIME = 12 # within a max of Y hours
"""
NOTES:
HadISD uses 2 days (48h)
Did try 7 days for initial run (October 2019) but still flagging lots of obs
Next version using 4 weeks (~1 month) to get the really isolated ones.
"""
MIN_SEPARATION = 28*24 # separated by Z hours on either side from other data


#*********************************************
def plot_cluster(times: np.array, obs_var: utils.Meteorological_Variable, oc_start: int, oc_end: int) -> None:
    '''
    Plot each odd cluster highlighted against surrounding data

    :param array times: datetime array
    :param MetVar obs_var: Meteorological variable object
    :param int oc_start: start of cluster in data array index
    :param int oc_end: end of cluster in data array index
    '''
    import matplotlib.pyplot as plt

    # sort the padding
    if oc_start == 0:
        start = 0
    else:
        start = oc_start - 20
        if start < 0:
            start = 0
    if oc_end == -1:
        end = -1
    else:
        end = oc_end + 20
        if end > len(times):
            end = len(times)

    # simple plot
    plt.clf()
    plt.plot(times[start: end], obs_var.data[start, end], 'bo')
    plt.plot(times[oc_start: oc_end], obs_var.data[oc_start: oc_end], 'ro')

    plt.ylabel(obs_var.name.capitalize())
    plt.show()

    return # plot_cluster

#************************************************************************
def flag_clusters(obs_var: utils.Meteorological_Variable, station: utils.Station,
                  plots: bool = False, diagnostics: bool = False) -> None:
    """
    Go through the clusters of data and flag if meet requirements

    :param MetVar obs_var: meteorological variable object
    :param Station station: Station Object for the station
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

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

                    if plots:
                        plot_cluster(station, obs_var, 0, cluster_end+1)

        elif ce == len(potential_cluster_ends) - 1:

            # check if cluster at end of series (long gap before last few points)
            cluster_length = station.times.iloc[-1] - station.times.iloc[cluster_end+1] # add one to find cluster start!
            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[cluster_end+1:]) < MAX_LENGTH_OBS:
                    flags[cluster_end+1:] = "o"

                    if plots:
                        plot_cluster(station, obs_var, cluster_end+1, -1)


        if ce > 0:
            # check for cluster in series.
            #  use previous gap > MIN_SEPARATION to define cluster and check length
            cluster_length = station.times.iloc[cluster_end] - station.times.iloc[potential_cluster_ends[ce-1]+1] # add one to find cluster start!
            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[potential_cluster_ends[ce-1]+1: cluster_end+1]) < MAX_LENGTH_OBS:
                    flags[potential_cluster_ends[ce-1]+1: cluster_end+1] = "o"

                    if plots:
                        plot_cluster(station.times, obs_var, potential_cluster_ends[ce-1]+1, cluster_end+1)

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Odd Cluster {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != "")[0])}")
    if diagnostics:

        print(f"Odd Cluster {obs_var.name}")
        print(f"   Cumulative number of flags set: {len(np.where(flags != "")[0])}")

    return # flag_clusters

#************************************************************************
def occ(station: utils.Station, var_list: list, config_file: str, full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
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
