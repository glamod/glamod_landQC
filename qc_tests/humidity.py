"""
Humidity Cross Checks
^^^^^^^^^^^^^^^^^^^^^

1. Check and flag instances of super saturation
2. Check and flag instances of dew point depression
"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
import datetime as dt

import qc_utils as utils


#************************************************************************
def super_saturation_check(temperatures, dewpoints, plots=False, diagnostics=False):
    """
    Flag locations where dewpoint is greater than air temperature

    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    sss, = np.ma.where(dewpoints.data > temperatures.data)

    flags[sss] = "H"

    # only flag the dewpoints
    dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)
        
    if diagnostics:
        
        print("Supersaturation {}".format(dewpoints.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # super_saturation_check

#************************************************************************
def dew_point_depression(temperatures, dewpoints, plots=False, diagnostics=False):
    """
    Flag locations where dewpoint equals air temperature

    :param MetVar temperatures: temperatures object
    :param MetVar dewpoints: dewpoints object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(temperatures.data.shape[0])])

    dpd = temperatures.data - dewpoints.data

    locs, = np.ma.where(dpd == 0)

    # TODO - decide on whether to only flag after a set number of instances.

    flags[locs] = "H"

    # only flag the dewpoints
    dewpoints.flags = utils.insert_flags(dewpoints.flags, flags)
        
    if diagnostics:
        
        print("Dewpoint Depression {}".format(dewpoints.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # dew_point_depression

#************************************************************************
def hcc(station, config_file, full=False, plots=False, diagnostics=False):
    """
    Extract the variables and pass to the Humidity Cross Checks

    :param Station station: Station Object for the station
    :param str configfile: string for configuration file (unused here)
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")

    # Super Saturation
    super_saturation_check(temperatures, dewpoints, plots=plots, diagnostics=diagnostics)

    # Dew Point Depression
    #    Note, won't have cloud-base or past-significant-weather
    #    Note, currently don't have precipitation information
    dew_point_depression(temperatures, dewpoints, plots=plots, diagnostics=diagnostics)

    # dew point cut-offs (HadISD) not run
    #  greater chance of removing good observations 
    #  18 July 2019 RJHD

    return # hcc

#************************************************************************
if __name__ == "__main__":
    
    print("humidity cross checks")
#************************************************************************
