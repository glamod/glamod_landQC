"""
Pressure Cross Checks
^^^^^^^^^^^^^^^^^^^^^

Check for observations where difference between station and sea level pressure
falls outside of the expected range.
"""
import sys
import numpy as np
import scipy as sp
import datetime as dt

import qc_utils as utils
#************************************************************************

# TODO - move threshold into a config file?
THRESHOLD = 4 # check if appropriate!

#************************************************************************
def pressure_offset(sealp, stnlp, plots=False, diagnostics=False):
    """
    Flag locations where difference between station and sea-level pressure
    falls outside of bounds

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(sealp.data.shape[0])])

    difference = sealp.data - stnlp.data

    average_difference = utils.average(difference)
    range_difference = utils.spread(difference)

    high, = np.ma.where(difference > (average_difference + THRESHOLD*range_difference))
    low, = np.ma.where(difference < (average_difference - THRESHOLD*range_difference))

    if len(high) != 0:
        flags[high] = "P"
        if diagnostics:
            print("Number of high differences {}".format(len(high)))
    if len(low) != 0:
        flags[low] = "P"
        if diagnostics:
            print("Number of low differences {}".format(len(low)))

    if plots:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()
        plt.hist(difference.compressed(), bins = np.arange(np.round(average_difference)-10, np.round(average_difference)+10, 0.1))
        plt.axvline(x = (average_difference + THRESHOLD*range_difference), ls = "--", c = "r")
        plt.axvline(x = (average_difference - THRESHOLD*range_difference), ls = "--", c = "r")
        plt.xlim([average_difference - 11, average_difference + 11])
        plt.ylabel("Observations")
        plt.xlabel("Difference (hPa)")
        plt.show()

    # only flag the dewpoints
    stnlp.flags = utils.insert_flags(stnlp.flags, flags)

    return # pressure_offset

#************************************************************************
def pcc(station, config_file, full=False, plots=False, diagnostics=False):
    """
    Extract the variables and pass to the Pressure Cross Checks

    :param Station station: Station Object for the station
    :param str configfile: string for configuration file (unused here)
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    sealp = getattr(station, "sea_level_pressure")
    stnlp = getattr(station, "station_level_pressure")

    pressure_offset(sealp, stnlp, plots=plots, diagnostics=diagnostics)

    return # pcc

#************************************************************************
if __name__ == "__main__":
    
    print("pressure cross checks")
#************************************************************************
