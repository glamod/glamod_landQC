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


def assess_start_cluster(station: utils.Station,
                         obs_var: utils.MeteorologicalVariable,
                         flags: np.ndarray,
                         cluster: np.ma.MaskedArray,
                         cluster_start: int,
                         cluster_end: int,
                         plots: bool = False) -> None:
    """Assess whether initial data points are an odd cluster

    Parameters
    ----------
    station : utils.Station
        Station to assess (for plotting)
    obs_var : utils.MeteorologicalVariable
        Met Var to assess
    flags : np.ndarray
        Array to hold the flags, if set
    cluster : np.ma.MaskedArray
        The masked times corresponding to the cluster
    cluster_start : int
        The index of the first cluster point
    cluster_end : int
        The index of the last cluster point
    plots : bool, optional
        Plot this cluster, by default False
    """

    # check if cluster at start of series (long gap after a first few points)
    cluster_length = cluster.compressed()[-1] - cluster.compressed()[0]

    if cluster_length/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
        # could be a cluster, pull out only data locations
        good_cluster_locs, = np.nonzero(cluster.mask == False)

        if len(flags[good_cluster_locs + cluster_start]) < MAX_LENGTH_OBS:
            flags[good_cluster_locs + cluster_start] = "o"

            if plots:
                plot_cluster(station.times, obs_var,
                             0, cluster_end+1)


def assess_mid_cluster(station: utils.Station,
                         obs_var: utils.MeteorologicalVariable,
                         flags: np.ndarray,
                         cluster: np.ma.MaskedArray,
                         cluster_start: int,
                         cluster_end: int,
                         plots: bool = False) -> None:
    """Assess whether data points are an odd cluster

    Parameters
    ----------
    station : utils.Station
        Station to assess (for plotting)
    obs_var : utils.MeteorologicalVariable
        Met Var to assess
    flags : np.ndarray
        Array to hold the flags, if set
    cluster : np.ma.MaskedArray
        The masked times corresponding to the cluster
    cluster_start : int
        The index of the first cluster point
    cluster_end : int
        The index of the last cluster point
    plots : bool, optional
        Plot this cluster, by default False
    """

    # And determine length from the compressed array (if single point, length == 0)
    cluster_length = cluster.compressed()[-1] - cluster.compressed()[0]

    if cluster_length/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
        # could be a cluster, pull out only data locations
        good_cluster_locs, = np.nonzero(cluster.mask == False)

        if len(flags[good_cluster_locs + cluster_start]) < MAX_LENGTH_OBS:
            flags[good_cluster_locs + cluster_start] = "o"

            if plots:
                plot_cluster(station.times, obs_var,
                             cluster_start, cluster_end+1)


def assess_end_cluster(station: utils.Station,
                       obs_var: utils.MeteorologicalVariable,
                       flags: np.ndarray,
                       cluster: np.ma.MaskedArray,
                       cluster_end: int,
                       plots: bool = False) -> None:
    """Assess whether final data points are an odd cluster

    Parameters
    ----------
    station : utils.Station
        Station to assess (for plotting)
    obs_var : utils.MeteorologicalVariable
        Met Var to assess
    flags : np.ndarray
        Array to hold the flags, if set
    cluster : np.ma.MaskedArray
        The masked times corresponding to the cluster
    cluster_end : int
        The final index of the previous cluster
    plots : bool, optional
        Plot this cluster, by default False
    """
    # And determine length from the compressed array (if single point, length == 0)
    cluster_length = cluster.compressed()[-1] - cluster.compressed()[0]

    if cluster_length/np.timedelta64(1, "h") < MAX_LENGTH_TIME:
        # could be a cluster, pull out only data locations
        good_cluster_locs, = np.nonzero(cluster.mask == False)

        if len(flags[good_cluster_locs + cluster_end + 1]) < MAX_LENGTH_OBS:
            flags[good_cluster_locs + cluster_end + 1] = "o"

            if plots:
                plot_cluster(station.times, obs_var, cluster_end, -1)


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
    good_locs, = np.nonzero(these_times.mask == False)
    time_differences = np.diff(these_times.compressed())/np.timedelta64(1, "m")

    potential_cluster_ends, = np.nonzero(time_differences >= MIN_SEPARATION * 60)


    if len(potential_cluster_ends) == 0:
        # no odd clusters identified
        logger.info(f"Odd Cluster {obs_var.name}")
        logger.info("   No flags set")
        return

    # spin through potential clusters
    for ce, cluster_end in enumerate(good_locs[potential_cluster_ends]):
        cluster_start = good_locs[potential_cluster_ends[ce-1]+1]

        if ce == 0:
            # Check for cluster right at the beginning of the series
            # Because of masks, pull data from data start through to end
            cluster = these_times[good_locs[0]: cluster_end+1]
            assess_start_cluster(station, obs_var, flags,
                                 cluster, good_locs[0], cluster_end,
                                 plots=plots)

        elif ce > 0:
            # Check for cluster in middle of series.

            # Because of masks, pull data from cluster start through to end
            cluster = these_times[cluster_start: cluster_end+1]
            assess_mid_cluster(station, obs_var, flags,
                               cluster, cluster_start, cluster_end,
                               plots=plots)


        # Additionally
        if ce == len(potential_cluster_ends) - 1:
            # Finally, check last stretch
            # As end of the sequence there's no end to calculate the time-diff for
            # check if cluster at end of series (long gap before last few points)
            cluster = these_times[cluster_end+1: ]
            assess_end_cluster(station, obs_var, flags,
                               cluster, cluster_end,
                               plots=plots)

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


    # occ

