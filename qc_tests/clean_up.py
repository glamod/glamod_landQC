"""
Clean Up Months
^^^^^^^^^^^^^^^

Flag whole months if large proportions already flagged, 
or if only a few observations left.
"""
#************************************************************************
import numpy as np

import qc_utils as utils

LOW_COUNT_THRESHOLD = 20
HIGH_FLAGGING_THRESHOLD = 0.4

#************************************************************************
def clean_up(obs_var, station, plots=False, diagnostics=False):
    """
    Check for high flagging rates within a calendar month and flag remaining

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    new_flags = np.array(["" for i in range(obs_var.data.shape[0])])

    old_flags = obs_var.flags

    for year in np.unique(station.years):

        for month in range(1, 13):

            month_locs, = np.where(np.logical_and(station.years == year, station.months == month))

            this_month = obs_var.data[month_locs]
            obs_locs, = np.where(this_month.mask == False)
            n_obs = np.ma.count(this_month)

            flagged, = np.where(old_flags[month_locs][obs_locs] != "")
            unflagged, = np.where(old_flags[month_locs][obs_locs] == "")

            if n_obs < LOW_COUNT_THRESHOLD:
                # insufficient unflagged observations left
                new_flags[month_locs[obs_locs]] = "E"
                if diagnostics:
                    print("{} - {} : {}".format(year, month, len(obs_locs)))

            else:
                if flagged.shape[0] / n_obs > HIGH_FLAGGING_THRESHOLD:
                    # flag remainder
                    new_flags[month_locs[obs_locs]] = "E"

                    if diagnostics:
                        print("{} - {} : {}".format(year, month, len(obs_locs)))

    if diagnostics:
        print("Clean Up {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(new_flags != "")[0])))

    return new_flags # clean_up

#************************************************************************
def mcu(station, var_list, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to monthly clean up

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        flags = clean_up(obs_var, station, plots=plots, diagnostics=diagnostics)

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    return # wrc

#************************************************************************
if __name__ == "__main__":

    print("removing remaining obs from highly flagged months")
#************************************************************************
