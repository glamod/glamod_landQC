"""
World Records Check
^^^^^^^^^^^^^^^^^^^

Check for exceedence of world records
"""
#************************************************************************
import numpy as np

import qc_utils as utils

#************************************************************************
# Fixed values at the moment
T_X = {"Africa" : 55.0, "Asia" : 53.9, "South_America" : 48.9, "North_America" : 56.7, "Europe" : 48.0, "Pacific" : 50.7, "Antarctica" : 15.0, "ROW" : 56.7}
T_N = {"Africa" : -23.9, "Asia" : -67.8, "South_America" : -32.8, "North_America" : -63.0, "Europe" : -58.1, "Pacific" : -23.0, "Antarctica" : -89.2, "ROW" : -89.2}
D_X = {"Africa" : 55.0, "Asia" : 53.9, "South_America" : 48.9, "North_America" : 56.7, "Europe" : 48.0, "Pacific" : 50.7, "Antarctica" : 15.0, "ROW" : 56.7}
D_N = {"Africa" : -50., "Asia" : -100., "South_America" : -60., "North_America" : -100., "Europe" : -100., "Pacific" : -50., "Antarctica" : -100., "ROW" : -100.}
W_X = {"Africa" : 113.2, "Asia" : 113.2, "South_America" : 113.2, "North_America" : 113.2, "Europe" : 113.2, "Pacific" : 113.2, "Antarctica" : 113.2, "ROW" : 113.2}
W_N = {"Africa" : 0., "Asia" : 0., "South_America" : 0., "North_America" : 0., "Europe" : 0., "Pacific" : 0., "Antarctica" : 0., "ROW" : 0.}
S_X = {"Africa" : 1083.3, "Asia" : 1083.3, "South_America" : 1083.3, "North_America" : 1083.3, "Europe" : 1083.3, "Pacific" : 1083.3, "Antarctica" : 1083.3, "ROW" : 1083.3}
S_N = {"Africa" : 870., "Asia" : 870., "South_America" : 870., "North_America" : 870., "Europe" : 870., "Pacific" : 870., "Antarctica" : 870., "ROW" : 870.}

# 
maxes = {"temperature" : T_X, "dew_point_temperature" : D_X, "wind_speed" : W_X, "sea_level_pressure" : S_X}
mins = {"temperature" : T_N, "dew_point_temperature" : D_N, "wind_speed" : W_N, "sea_level_pressure" : S_N}


#************************************************************************
def record_check(obs_var, plots=False, diagnostics=False):
    """
    Check for exceedences of world record values

    :param MetVar obs_var: meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # presume ROW at the moment until have a regional assignator
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    # masked arrays allows ignoring missing data
    too_high, = np.ma.where(obs_var.data > maxes[obs_var.name]["ROW"])
    too_low, = np.ma.where(obs_var.data < mins[obs_var.name]["ROW"])

    flags[too_high] = "W"
    flags[too_low] = "W"

    if diagnostics:
        print("World Records {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return flags # record_check

#************************************************************************
def wrc(station, var_list, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the World Record Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        flags = record_check(obs_var, plots=plots, diagnostics=diagnostics)

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    return # wrc

#************************************************************************
if __name__ == "__main__":

    print("checking for exceedence of world records")
#************************************************************************
