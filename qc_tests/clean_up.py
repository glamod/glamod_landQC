"""
Clean Up Months
^^^^^^^^^^^^^^^

Flag whole months if large proportions already flagged, 
or if only a few observations left [not active].
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils

LOW_COUNT_THRESHOLD = 0
HIGH_FLAGGING_THRESHOLD = 0.6

#************************************************************************
def clean_up(obs_var: utils.Meteorological_Variable, station: utils.Station,
             low_counts: int = LOW_COUNT_THRESHOLD, high_flags: float = HIGH_FLAGGING_THRESHOLD,
             plots: bool = False, diagnostics:bool = False) -> np.ndarray:
    """
    Check for high flagging rates within a calendar month and flag remaining

    :param MetVar obs_var: meteorological variable object
    :param Station station: Station object
    :param int low_counts: threshold of low counts below which remaining unflagged obs are flagged
    :param float high_flags: threshold above which flags are set on remaining unflagged obs
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

            if unflagged.shape[0] < low_counts:
                # insufficient unflagged observations left
                new_flags[month_locs[obs_locs][unflagged]] = "E"
                logger.info(f"Low count {year} - {month} : {len(obs_locs)}")

            else:
                if flagged.shape[0] == 0:
                    # no flags set so just skip
                    pass
                elif flagged.shape[0] / n_obs > high_flags:
                    # flag remainder
                    new_flags[month_locs[obs_locs][unflagged]] = "E"
                    if diagnostics:
                        print(f"Clean up high flagging {year} - {month} : {len(obs_locs)} ({(100*flagged.shape[0] / n_obs)}%)")


    logger.info(f"Clean Up {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(new_flags != '')[0])}")

    return new_flags # clean_up

#************************************************************************
def mcu(station: utils.Meteorological_Variable, var_list: list, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
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
