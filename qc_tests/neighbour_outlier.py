'''
Neighbour Outlier Check
=======================

Check for timestamps where difference to sufficient fraction of neighbours is
sufficiently high.
'''
#************************************************************************
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)

# internal utils
import qc_utils as utils
import io_utils
import setup
#************************************************************************

MIN_SPREAD = 2 # TODO should this be different for each variable?
SPREAD_LIMIT = 5 # matches HadISD which used 5*IQR
DUBIOUS_FRACTION = 2./3. # more than 2/3 of neighbours suggest differences dubious
STORM_FRACTION = 2./3. # fraction of negative differences to target (likely at centre of low pressure system)
DISTANT_NEIGHBOURS = 100 # km

#************************************************************************
def plot_neighbour_flags(times: np.ndarray, flagged_time: int,
                         target: utils.MeteorologicalVariable, buddies: np.ndarray) -> None:
    '''
    Plot each spike against surrounding data

    :param array times: datetime array
    :param int flagged_time: location of flagged value
    :param MetVar target: meteorological variable to plot
    :param array buddies: 2d array of all buddy's data
    '''
    import matplotlib.pyplot as plt

    # simple plot
    plt.clf()
    pad_start = flagged_time-48
    if pad_start < 0:
        pad_start = 0
    pad_end = flagged_time+48
    if pad_end > len(target.data):
        pad_end = len(target.data)

    plt.plot(times[pad_start: pad_end], target.data[pad_start: pad_end], 'k-', marker="o", zorder=5)

    for buddy in buddies:
        plt.plot(times[pad_start: pad_end], buddy[pad_start: pad_end], c='0.5', ls="-", marker=".")

    plt.plot(times[flagged_time], target.data[flagged_time], 'r*', ms=10, zorder=10)

    plt.ylabel(target.name.capitalize())
    plt.show()

    return # plot_neighbour_flags

#************************************************************************
def read_in_buddies(target_station: utils.Station, initial_neighbours: np.ndarray,
                    diagnostics: bool = False, plots: bool = False) -> dict:
    """
    Read in the buddy data for the neighbours, and store in dictionary

    :param Station target_station: station to run on
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test

    :returns: dict of the buddy objects

    """
    # Get dataframe of all sations
    station_list = utils.get_station_list()

    all_buddies = {}

    for bid, buddy_id in enumerate(initial_neighbours[:, 0]):

        if buddy_id == target_station.id:
            # first entry is self
            continue
        if buddy_id == "-":
            # end of the list of buddies
            break

        if diagnostics:
            print(f"Buddy number {bid+1}/{len(initial_neighbours[:, 0])} {buddy_id}")

        # set up station object to hold information
        buddy_idx, = np.where(station_list.id == buddy_id)
        buddy = utils.Station(buddy_id, station_list.iloc[buddy_idx].latitude.values[0],
                                station_list.iloc[buddy_idx].longitude.values[0],
                                station_list.iloc[buddy_idx].elevation.values[0])

        try:
            buddy, _ = io_utils.read_station(os.path.join(setup.SUBDAILY_PROC_DIR,
                                              f"{buddy_id:11s}.qff{setup.OUT_COMPRESSION}"),
                                              buddy, read_flags=True)

            # store in dictionary, so that can ensure correct look up
            all_buddies[buddy_id] = buddy

        except OSError: # as e:
            # file missing, move on to next in sequence
            io_utils.write_error(target_station, f"File Missing (Buddy check): {buddy_id}")
            continue
        except ValueError as e:
            # some issue in the raw file
            io_utils.write_error(target_station, f"Error in input file (Buddy check): {buddy_id}", error=str(e))
            continue

    logger.info("All buddies read in")

    return all_buddies


def read_in_buddy_data(target_station: utils.Station, initial_neighbours: np.ndarray,
                       all_buddies: dict, variable: utils.MeteorologicalVariable,
                       diagnostics: bool = False, plots: bool = False) -> np.ndarray:
    """
    Read in the buddy data for the neighbours for specified variable

    :param Station target_station: station to run on
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param dict all_buddies: container of all the buddy objects, indexed by ID
    :param str variable: obs variable being run on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test

    :returns: array of the buddy data [first line corresponds to target, so empty]

    """

    #*************************
    # read in in the neighbour (buddy) data
    all_buddy_data = np.ma.zeros([len(initial_neighbours[:, 0]), len(target_station.times)])
    all_buddy_data.mask = np.ones(all_buddy_data.shape)

    buddy_count = 0
    for bid, buddy_id in enumerate(initial_neighbours[:, 0]):
        if buddy_id == target_station.id:
            # first entry is self
            continue
        if buddy_id == "-":
            # end of the list of buddies
            break

        if diagnostics:
            print(f"Buddy number {bid+1}/{len(initial_neighbours[:, 0])} {buddy_id}")

        # If buddy missing, then no key in the dictionary
        #   Capture and move on, as KeyError below results in confusing message
        if buddy_id not in all_buddies.keys():
            if diagnostics:
                print(f"  No entry for Buddy, file missing")
            continue

        try:
            buddy = all_buddies[buddy_id]

            buddy_var = getattr(buddy, variable)

            # apply flags
            flag_locs, = np.nonzero(buddy_var.flags != "")
            buddy_var.data.mask[flag_locs] = True

        except KeyError as e:
            # file missing, move on to next in sequence
            io_utils.write_error(target_station, f"Entry Missing (Buddy check): {variable}) - {buddy_id}", error=str(e))
            continue
        except AttributeError as e:
            # file missing, move on to next in sequence
            io_utils.write_error(target_station, f"Variable Missing (Buddy check): {variable} - {buddy_id}", error=str(e))
            continue
        # match the timestamps of target_station and copy over
        # TODO: use pandas routine: target_station.times.isin(buddy.times)
        match = np.in1d(target_station.times, buddy.times)
        match_back = np.in1d(buddy.times, target_station.times)

        if True in match and True in match_back:
            # skip if no overlapping times at all!
            all_buddy_data[bid, match] = buddy_var.data[match_back]
            buddy_count += 1

    if buddy_count < utils.MIN_NEIGHBOURS:
        logger.warning(f"Insufficient number of neighbours with data for {variable} ({buddy_count} < {utils.MIN_NEIGHBOURS})")
    else:
        logger.info(f"Number of neighbours with data for {variable}: {buddy_count}")

    return all_buddy_data


#************************************************************************
def calculate_data_spread(target_station: utils.Station,
                          differences: np.ma.MaskedArray) -> np.ndarray:
    """
    Find spread of differences on monthly basis (with minimum value)

    :param Station target_station: station to run on
    :param np.ma.MaskedArray differences: array of differences of buddy data to target data

    :returns: np.ndarray of the spread of the differences
    """

    spreads = np.ma.zeros(differences.shape)

    for month in range(1, 13):
        # note this takes _all_ years for the target-buddy pair for this calendar month
        month_locs, = np.nonzero(target_station.months == month)

        for bid, buddy in enumerate(differences):

            if len(buddy[month_locs].compressed()) > utils.DATA_COUNT_THRESHOLD:
                this_spread = utils.spread(buddy[month_locs])
                if this_spread < MIN_SPREAD:
                    spreads[bid, month_locs] = MIN_SPREAD
                else:
                    spreads[bid, month_locs] = this_spread

            else:
                spreads[bid, month_locs] = MIN_SPREAD

    spreads.mask = np.copy(differences.mask)
    return spreads


#************************************************************************
def adjust_pressure_for_tropical_storms(dubious: np.ma.MaskedArray, initial_neighbours: np.ndarray,
                                        differences: np.ndarray, spreads: np.ndarray) -> np.ndarray:
    """
    For pressure variables look for tropical storm passages, where large differences
    of station under the "eye" to neighbours are real and expected.

    :param np.ma.MaskedArray dubious: locations where differences are greater than expected
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param array differences: array of differences between target and neighbours
    :param array spreads: array of the spread of the differences

    :returns: np.ndarray of locations where target values are dubious given the neighbours
    """
    # select on distance
    distant, = np.where(initial_neighbours[:, 1].astype(int) > DISTANT_NEIGHBOURS)

    if len(distant) > 0:
        # find positive and negative differences across neighbours
        positive = np.ma.where(differences[distant] > spreads[distant]*SPREAD_LIMIT)
        negative = np.ma.where(differences[distant] < -spreads[distant]*SPREAD_LIMIT)
        # as 2-d array, positive/negative are tuples of (station, timestamp) locations

        # spin through each neighbour
        for dn, dist_neigh in enumerate(distant):
            # pull out the large differences corresponding to each neighbour
            pos, = np.where(positive[0] == dn)
            neg, = np.where(negative[0] == dn)

            # by default, flag all dubious values greater difference than expected
            dubious[dist_neigh, positive[1][pos]] = 1
            dubious[dist_neigh, negative[1][neg]] = 1

            # but if there are negative ones, then if these dominate,
            #   it's likely because of storms, so unflag these as we want to retain
            if len(neg) > 0:
                fraction = len(neg)/(len(pos) + len(neg))
                if fraction > STORM_FRACTION:
                    # majority negative across matching record
                    #   only flag the positives [definitely not storms]
                    dubious[dist_neigh, negative[1][neg]] = 0

    else:
        # all stations close by so storms shouldn't affect, include all
        # note where differences exceed the spread (using np.abs to get all)
        dubious_locs = np.ma.where(np.ma.abs(differences) > spreads*SPREAD_LIMIT)
        dubious[dubious_locs] = 1

    return dubious


#************************************************************************
def neighbour_outlier(target_station: utils.Station, initial_neighbours: np.ndarray,
                      all_buddies: dict, variable: utils.MeteorologicalVariable,
                      diagnostics: bool = False,
                      plots: bool = False, full: bool = False) -> None:
    """
    Works on a single station and variable.  Reads in neighbour's data,
    finds locations where sufficent are sufficiently different.

    :param Station target_station: station to run on
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param dict all_buddies: container for all the buddy objects, indexed by ID
    :param str variable: obs variable being run on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    """

    #*************************
    # extract target observations
    obs_var = getattr(target_station, variable)
    flags = np.array(["" for i in range(obs_var.data.shape[0])]).astype("<U10")

    #*************************
    # Read in the buddy data
    all_buddy_data = read_in_buddy_data(target_station, initial_neighbours, all_buddies,
                                        variable, diagnostics=diagnostics, plots=plots)

    #*************************
    # find differences
    differences = obs_var.data - all_buddy_data

    #*************************
    # find spread of differences on monthly basis (with minimum value)
    spreads = calculate_data_spread(target_station, differences)

    # store which entries may be sufficient to flag
    dubious = np.ma.zeros(differences.shape)
    dubious.mask = np.copy(differences.mask)

    #*************************
    # adjust for storms
    if variable in ["sea_level_pressure", "station_level_pressure"]:
        dubious = adjust_pressure_for_tropical_storms(dubious, initial_neighbours,
                                                      differences, spreads)
    else:
        #*************************
        # note where differences exceed the spread [all non pressure variables]
        dubious_locs = np.ma.where(np.ma.abs(differences) > spreads*SPREAD_LIMIT)
        dubious[dubious_locs] = 1

    logger.info("cross checks complete - assessing all outcomes")

    #*************************
    # sum across neighbours
    neighbour_count = np.ma.count(differences, axis=0)
    dubious_count = np.ma.sum(dubious, axis=0)

    # flag if large enough fraction (>0.66)
    sufficient, = np.ma.where(dubious_count > DUBIOUS_FRACTION*neighbour_count)

    flags[sufficient] = "N"

    if plots:
        for flag in sufficient:
            plot_neighbour_flags(target_station.times, flag, obs_var, all_buddy_data)

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Neighbour Outlier {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # neighbour_outlier

#************************************************************************
def noc(target_station: utils.Station, initial_neighbours: np.ndarray, var_list: list,
        full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Neighbour Outlier Check

    :param Station station: Station Object for the station
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param list var_list: list of variables to test
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # Check if have sufficient neighbours, as if too few, might as well exit here.
    #    First entry is self (target station), hence the "-1" at the end.
    n_neighbours = len(np.where(initial_neighbours[:, 0] != "-")[0]) - 1
    if n_neighbours < utils.MIN_NEIGHBOURS:
        logger.warning(f"{target_station.id} has insufficient neighbours ({n_neighbours}<{utils.MIN_NEIGHBOURS})")
        if diagnostics:
            print(f"{target_station.id} has insufficient neighbours ({n_neighbours}<{utils.MIN_NEIGHBOURS})")

    else:
        logger.info(f"{target_station.id} has {n_neighbours} neighbours")

        # read in neighbour stations and store.
        all_buddies = read_in_buddies(target_station, initial_neighbours,
                                      diagnostics=diagnostics, plots=plots)
        for var in var_list:
            neighbour_outlier(target_station, initial_neighbours, all_buddies, var,
                              diagnostics=diagnostics, plots=plots, full=full)

    # noc

