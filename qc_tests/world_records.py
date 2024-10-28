"""
World Records Check
^^^^^^^^^^^^^^^^^^^

Check for exceedence of world records

Uses values at https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
and extra knowledge where available
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils

#************************************************************************
# Fixed values at the moment.

# row = Rest Of World.
# updated from https://wmo.asu.edu/content/world-meteorological-organization-global-weather-climate-extremes-archive
#   augmented by local knowledge from extreme events in these cases:
#   T_X : europe : 48.8 (Syracuse, Sicily, August 2021) beating previous 48.0 [WMO still to validate]

# last check 27 Jul 2022, RJHD

T_X = {"africa" : 55.0, "asia" : 53.9, "samerica" : 48.9, "namerica" : 56.7, "europe" : 48.8,
       "oceania" : 50.7, "antarctica" : 18.3, "row" : 56.7}
T_N = {"africa" : -23.9, "asia" : -67.8, "samerica" : -32.8, "namerica" : -63.0, "europe" : -58.1,
       "oceania" : -23.0, "antarctica" : -89.2, "row" : -89.2}
D_X = {"africa" : 55.0, "asia" : 53.9, "samerica" : 48.9, "namerica" : 56.7, "europe" : 48.0,
       "oceania" : 50.7, "antarctica" : 18.3, "row" : 56.7}
D_N = {"africa" : -50., "asia" : -100., "samerica" : -60., "namerica" : -100., "europe" : -100.,
       "oceania" : -50., "antarctica" : -100., "row" : -100.}
W_X = {"africa" : 113.2, "asia" : 113.2, "samerica" : 113.2, "namerica" : 113.2, "europe" : 113.2,
       "oceania" : 113.2, "antarctica" : 113.2, "row" : 113.2}
W_N = {"africa" : 0., "asia" : 0., "samerica" : 0., "namerica" : 0., "europe" : 0., "oceania" : 0.,
       "antarctica" : 0., "row" : 0.}
S_X = {"africa" : 1083.3, "asia" : 1083.3, "samerica" : 1083.3, "namerica" : 1083.3, "europe" : 1083.3,
       "oceania" : 1083.3, "antarctica" : 1083.3, "row" : 1083.3}
S_N = {"africa" : 870., "asia" : 870., "samerica" : 870., "namerica" : 870., "europe" : 870.,
       "oceania" : 870., "antarctica" : 870., "row" : 870.}

# 
maxes = {"temperature" : T_X, "dew_point_temperature" : D_X, "wind_speed" : W_X, "sea_level_pressure" : S_X}
mins = {"temperature" : T_N, "dew_point_temperature" : D_N, "wind_speed" : W_N, "sea_level_pressure" : S_N}

  

#************************************************************************
def record_check(obs_var: utils.Meteorological_Variable, continent: str,
                 plots: bool = False, diagnostics: bool = False) -> np.ndarray:
    """
    Check for exceedences of world record values

    :param MetVar obs_var: meteorological variable object
    :param str continent: continent of the station location
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    :returns: np.array of flags
    """
    assert isinstance(obs_var, utils.Meteorological_Variable)
    assert isinstance(continent, str)
    
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    if continent in ["africa", "asia", "samerica", "namerica", "europe", "oceania", "antarctica", "row"]:
        # masked arrays allows ignoring missing data
        too_high, = np.ma.where(obs_var.data > maxes[obs_var.name][continent])
        too_low, = np.ma.where(obs_var.data < mins[obs_var.name][continent])

    else:
        # Use ROW for "none" or any other result.
        # masked arrays allows ignoring missing data
        too_high, = np.ma.where(obs_var.data > maxes[obs_var.name]["row"])
        too_low, = np.ma.where(obs_var.data < mins[obs_var.name]["row"])

    flags[too_high] = "W"
    flags[too_low] = "W"

    logger.info(f"World Records {obs_var.name} ({continent})")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return flags # record_check

#************************************************************************
def wrc(station: utils.Station, var_list: list, full: bool = False,
        plots: bool = False,diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the World Record Check.

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param bool full: run a full update (unused here)
    :param book plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        flags = record_check(obs_var, 
                             station.continent, 
                             plots=plots, 
                             diagnostics=diagnostics
        )

        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    return # wrc

#************************************************************************
if __name__ == "__main__":

    print("checking for exceedence of world records")
#************************************************************************
