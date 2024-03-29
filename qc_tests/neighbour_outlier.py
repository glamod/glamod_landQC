'''
Neighbour Outlier Check
^^^^^^^^^^^^^^^^^^^^^^^

Check for timestamps where difference to sufficient fraction of neighbours is
sufficiently high
'''
#************************************************************************
import os
import datetime as dt
import pandas as pd
import numpy as np

# internal utils
import qc_utils as utils
import io_utils as io
import qc_tests
import setup
#************************************************************************

MIN_SPREAD = 2 # TODO should this be different for each variable?
SPREAD_LIMIT = 5 # matches HadISD which used 5*IQR


#************************************************************************
def plot_neighbour_flags(times, flagged_time, target, buddies):
    '''
    Plot each spike against surrounding data

    :param array times: datetime array
    :param MetVar target: Meteorological variable object
    :param int spike_start: the location of the spike
    :param int spike_length: the length of the spike

    :returns:
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
def neighbour_outlier(target_station, initial_neighbours, variable, diagnostics=False, plots=False, full=False):
    """
    Works on a single station and variable.  Reads in neighbour's data, finds locations where sufficent are sufficiently different.

    :param Station target_station: station to run on 
    :param array initial_neighbours: input neighbours (ID, distance) pairs
    :param str variable: obs variable being run on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    """
    station_list = utils.get_station_list()

    # if sufficient
    n_neighbours = len(np.where(initial_neighbours[:, 0] != "-")[0])-1
    if n_neighbours < utils.MIN_NEIGHBOURS:
        print("{} has insufficient neighbours ({}<{})".format(target_station.id, n_neighbours, utils.MIN_NEIGHBOURS))

    else:
        #*************************
        # extract target observations
        obs_var = getattr(target_station, variable)
        flags = np.array(["" for i in range(obs_var.data.shape[0])]).astype("<U10")

        #*************************
        # read in in the neighbour (buddy) data
        all_buddy_data = np.ma.zeros([len(initial_neighbours[:, 0]), len(target_station.times)])
        all_buddy_data.mask = np.ones(all_buddy_data.shape)

        for bid, buddy_id in enumerate(initial_neighbours[:, 0]):
            if buddy_id == target_station.id:
                # first entry is self
                continue
            if buddy_id == "-":
                # end of the list of buddies
                break

            if diagnostics:
                print("{}/{} {}".format(bid, len(initial_neighbours[:, 0]), buddy_id))

            # set up station object to hold information
            buddy_idx, = np.where(station_list.id == buddy_id)
            buddy = utils.Station(buddy_id, station_list.iloc[buddy_idx].latitude.values[0], \
                                      station_list.iloc[buddy_idx].longitude.values[0], station_list.iloc[buddy_idx].elevation.values[0])

            try:
                buddy, buddy_df = io.read_station(os.path.join(setup.SUBDAILY_PROC_DIR, "{:11s}.qff{}".format(buddy_id, setup.OUT_COMPRESSION)), buddy, read_flags=True) 

                buddy_var = getattr(buddy, variable)

                # apply flags
                flag_locs, = np.where(buddy_var.flags != "") 
                buddy_var.data.mask[flag_locs] = True

            except OSError as e:
                # file missing, move on to next in sequence
                io.write_error(target_station, "File Missing (Buddy, {}) - {}".format(variable, buddy_id))
                continue
            except ValueError as e:
                # some issue in the raw file
                io.write_error(target_station, "Error in input file (Buddy, {}) - {}".format(variable, buddy_id), error=str(e))
                continue

            # match the timestamps of target_station and copy over
            match = np.in1d(target_station.times, buddy.times) 
            match_back = np.in1d(buddy.times, target_station.times) 

            if True in match and True in match_back:
                # skip if no overlapping times at all!
                all_buddy_data[bid, match] = buddy_var.data[match_back]


        if diagnostics:
            print("All buddies read in")
                    
        #*************************
        # find differences
        differences = all_buddy_data - obs_var.data

        #*************************
        # find spread of differences on monthly basis (with minimum value)
        spreads = np.ma.zeros(differences.shape)

        for month in range(1, 13):

            month_locs = np.where(target_station.months == month)

            for bid, buddy in enumerate(differences):

                if len(differences[bid, month_locs].compressed()) > utils.DATA_COUNT_THRESHOLD:

                    this_spread = utils.spread(differences[bid, month_locs])
                    if this_spread < MIN_SPREAD:
                        spreads[bid, month_locs] = MIN_SPREAD
                    else:
                        spreads[bid, month_locs] = utils.spread(differences[bid, month_locs])

                else:
                    spreads[bid, month_locs] = MIN_SPREAD

        spreads.mask = np.copy(differences.mask)

        # store which entries may be sufficient to flag
        dubious = np.ma.zeros(differences.shape)
        dubious.mask = np.copy(differences.mask)

        #*************************
        # adjust for storms
        if variable in ["sea_level_pressure", "station_level_pressure"]:
            distant, = np.where(initial_neighbours[:, 1].astype(int) > 100)
            if len(distant) > 0:
                # find positive and negative differences across neighbours
                positive = np.ma.where(differences[distant] > spreads[distant]*SPREAD_LIMIT)
                negative = np.ma.where(differences[distant] < spreads[distant]*SPREAD_LIMIT)

                # spin through each neighbour
                for dn, dist_neigh in enumerate(distant):

                    pos, = np.where(positive[0] == dn)
                    neg, = np.where(negative[0] == dn)

                    if len(neg) > 0:
                        ratio = len(neg)/(len(pos) + len(neg))
                        if ratio > 0.667:
                            # majority negative, only flag the positives [definitely not storms]
                            dubious[dist_neigh, positive[1][pos]] = 1

            else:
                # all stations close by so storms shouldn't affect, include all
                # note where differences exceed the spread
                dubious_locs = np.ma.where(np.ma.abs(differences) > spreads*SPREAD_LIMIT)
                dubious[dubious_locs] = 1

        else:
            #*************************
            # note where differences exceed the spread [all non pressure variables]
            dubious_locs = np.ma.where(np.ma.abs(differences) > spreads*SPREAD_LIMIT)
            dubious[dubious_locs] = 1


        if diagnostics:
            print("cross checks complete - assessing all outcomes")
        #*************************
        # sum across neighbours
        neighbour_count = np.ma.count(differences, axis=0)
        dubious_count = np.ma.sum(dubious, axis=0)

        # flag if large enough fraction (>0.66)
        sufficient, = np.ma.where(dubious_count > 0.66*neighbour_count)
        flags[sufficient] = "N"

        if plots:
            for flag in sufficient:
                plot_neighbour_flags(target_station.times, flag, obs_var, all_buddy_data)


        # append flags to object
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        if diagnostics:

            print("Neighbour Outlier {}".format(obs_var.name))
            print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # neighbour_outlier

#************************************************************************
def noc(target_station, initial_neighbours, var_list, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Neighbour Outlier Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        neighbour_outlier(target_station, initial_neighbours, var, diagnostics=diagnostics, plots=plots, full=full)

    return # noc

#************************************************************************
if __name__ == "__main__":

    print("checking for outliers compared to neighbours")
#************************************************************************
