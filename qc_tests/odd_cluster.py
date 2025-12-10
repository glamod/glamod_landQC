"""
Odd Cluster Checks
==================

Find isolated data (single points and small runs)
"""
#************************************************************************
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils
#************************************************************************

MAX_LENGTH_OBS = 6 # up to X data points
MAX_LENGTH_TIME = 12 # within a max of Y hours
"""
NOTES:
HadISD uses 2 days (48h)
Did try 7 days for initial run (October 2019) but still flagging lots of obs
Next version using 4 weeks (~1 month) to get the really isolated ones.
Release8.1 changed to 2 weeks (14days) as perhaps a better balance on test station (AJM00037849)
   and now set in configuration file
"""
MIN_SEPARATION = utils.ODD_CLUSTER_SEPARATION * 24
# separated by Z days on either side from other data


#*********************************************
def plot_cluster(times: pd.Series, obs_var: utils.MeteorologicalVariable,
                 oc_start: int, oc_end: int | None) -> None:  # pragma: no cover
    '''
    Plot each odd cluster highlighted against surrounding data

    :param Series times: datetime array
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
        oc_end = None
    else:
        end = oc_end + 20
        if end > len(times):
            end = len(times)

    # simple plot
    plt.clf()
    plt.plot(times[start: end], obs_var.data[start: end], 'bo')
    plt.plot(times[oc_start: oc_end], obs_var.data[oc_start: oc_end], 'ro')

    plt.ylabel(obs_var.name.capitalize())
    plt.show()

    # plot_cluster

#************************************************************************
def flag_clusters(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                  plots: bool = False, diagnostics: bool = False) -> None:
    """
    Go through the clusters of data and flag if meet requirements

    :param MetVar obs_var: meteorological variable object
    :param Station station: Station Object for the station
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    these_times = np.ma.copy(station.times)
    these_times.mask = obs_var.data.mask
    time_differences = np.ma.diff(these_times)/np.timedelta64(1, "m")

    potential_cluster_ends, = np.nonzero(time_differences >= MIN_SEPARATION * 60)

    if len(potential_cluster_ends) == 0:
        # no odd clusters identified
        logger.info(f"Odd Cluster {obs_var.name}")
        logger.info("   No flags set")
        return


    # spin through the *end*s of potential clusters
    for ce, cluster_end in enumerate(potential_cluster_ends):
        if ce == 0:
            # check if cluster at start of series (long gap after a first few points)
            cluster_length = station.times.iloc[cluster_end]-station.times.iloc[0]

            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[:cluster_end+1]) < MAX_LENGTH_OBS:
                    flags[:cluster_end+1] = "o"

                    if plots:
                        plot_cluster(station.times, obs_var, 0, cluster_end+1)

        elif ce > 0:
            # check for cluster in series.
            #  use previous gap > MIN_SEPARATION to define cluster and check length
            cluster_length = station.times.iloc[cluster_end] - station.times.iloc[potential_cluster_ends[ce-1]+1] # add one to find cluster start!

            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[potential_cluster_ends[ce-1]+1: cluster_end+1]) < MAX_LENGTH_OBS:
                    flags[potential_cluster_ends[ce-1]+1: cluster_end+1] = "o"

                    if plots:
                        plot_cluster(station.times, obs_var, potential_cluster_ends[ce-1]+1, cluster_end+1)

        if ce == len(potential_cluster_ends) - 1:
            # Finally, check last stretch
            # As end of the sequence there's no end to calculate the time-diff for
            # check if cluster at end of series (long gap before last few points)
            cluster_length = station.times.iloc[-1] - station.times.iloc[cluster_end+1] # add one to find cluster start!

            if cluster_length.asm8/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
                # could be a cluster
                if len(flags[cluster_end+1:]) < MAX_LENGTH_OBS:
                    flags[cluster_end+1:] = "o"

                    if plots:
                        plot_cluster(station.times, obs_var, cluster_end+1, -1)

    # append flags to object
    obs_var.store_flags(utils.insert_flags(obs_var.flags, flags))

    logger.info(f"Odd Cluster {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # flag_clusters


#************************************************************************
def occ(station: utils.Station, var_list: list, config_dict: dict,
        full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Odd Cluster Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param dict config_dict: dictionary for settings (unused at the moment)
    :param bool full: run a full update (unused at the moment)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        flag_clusters(obs_var, station, plots=plots, diagnostics=diagnostics)


    # dgc


