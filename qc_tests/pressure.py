"""
Pressure Cross Checks
^^^^^^^^^^^^^^^^^^^^^

Check for observations where difference between station and sea level pressure
falls outside of the expected range.
"""
import numpy as np

import qc_utils as utils
#************************************************************************

# TODO - move threshold into a config file?
THRESHOLD = 4 # min spread of 1hPa, so only outside +/-4hPa flagged.

MIN_OBS = 10

#*********************************************
def plot_pressure(sealp, stnlp, times, bad):
    '''
    Plot each observation of SSS or DPD against surrounding data

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param int bad: the location of SSS or DPD

    :returns:
    '''
    import matplotlib.pyplot as plt
    
    pad_start = bad - 24
    if pad_start < 0:
        pad_start = 0
    pad_end = bad + 24
    if pad_end > len(obs_var.data):
        pad_end = len(obs_var.data)
    
    # simple plot
    plt.clf()
    plt.plot(times[pad_start : pad_end], sealp.data[pad_start : pad_end], 'k-', marker=".", label=sealp.name.capitalize())
    plt.plot(times[pad_start : pad_end], stnlp.data[pad_start : pad_end], 'b-', marker=".", label=stnlp.name.capitalize())
    plt.plot(times[bad], sealp.data[bad], 'r*', ms=10)    
    plt.plot(times[bad], stnlp.data[bad], 'r*', ms=10)    

    plt.legend(loc = "upper right")
    plt.ylabel(sealp.units)

    plt.show()

    return # plot_pressure

#************************************************************************
def pressure_offset(sealp, stnlp, times, plots=False, diagnostics=False):
    """
    Flag locations where difference between station and sea-level pressure
    falls outside of bounds

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(sealp.data.shape[0])])

    difference = sealp.data - stnlp.data

    if len(difference.compressed()) >= MIN_OBS:

        average_difference = utils.average(difference)
        range_difference = utils.spread(difference)
        if range_difference < 1.0: # less than 1hPa
            range_difference = 1.0

        high, = np.ma.where(difference > (average_difference + (THRESHOLD*range_difference)))
        low, = np.ma.where(difference < (average_difference - (THRESHOLD*range_difference)))

        # diagnostic plots
        if plots:
            bins = np.arange(np.round(average_difference)-1, np.round(average_difference)+1, 0.1)
            import matplotlib.pyplot as plt
            plt.clf()
            plt.hist(difference.compressed(), bins=bins)
            plt.axvline(x=(average_difference + (THRESHOLD*range_difference)), ls="--", c="r")
            plt.axvline(x=(average_difference - (THRESHOLD*range_difference)), ls="--", c="r")
            plt.xlim([bins[0] - 1, bins[-1] + 1])
            plt.ylabel("Observations")
            plt.xlabel("Difference (hPa)")
            plt.show()

        if len(high) != 0:
            flags[high] = "p"
            if diagnostics:
                print("Pressure".format(stnlp.name))
                print("   Number of high differences {}".format(len(high)))
            if plots:
                for bad in high:
                    plot_pressure(sealp, stnlp, times, bad)

        if len(low) != 0:
            flags[low] = "p"
            if diagnostics:
                print("   Number of low differences {}".format(len(low)))
            if plots:
                for bad in low:
                    plot_pressure(sealp, stnlp, times, bad)


        # only flag the station level pressure
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

    pressure_offset(sealp, stnlp, station.times, plots=plots, diagnostics=diagnostics)

    return # pcc

#************************************************************************
if __name__ == "__main__":
    
    print("pressure cross checks")
#************************************************************************
