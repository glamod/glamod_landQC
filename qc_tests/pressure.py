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
THEORY_THRESHOLD = 15 # allows for small offset as well as intrinsic spread

MIN_SPREAD = 1.0
MAX_SPREAD = 5.0


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
    if pad_end > len(sealp.data):
        pad_end = len(sealp.data)

    # simple plot
    plt.clf()
    plt.plot(times[pad_start : pad_end], sealp.data[pad_start : pad_end], 'k-', marker=".", label=sealp.name.capitalize())
    plt.plot(times[pad_start : pad_end], stnlp.data[pad_start : pad_end], 'b-', marker=".", label=stnlp.name.capitalize())
    plt.plot(times[bad], sealp.data[bad], 'r*', ms=10)
    plt.plot(times[bad], stnlp.data[bad], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(sealp.units)

    plt.show()

    return # plot_pressure

#************************************************************************
def identify_values(sealp, stnlp, times, config_file, plots=False, diagnostics=False):
    """
    Find average and spread of differences

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param str configfile: string for configuration file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    difference = sealp.data - stnlp.data

    if len(difference.compressed()) >= utils.DATA_COUNT_THRESHOLD:

        average = utils.average(difference)
        spread = utils.spread(difference)
        if spread < MIN_SPREAD: # less than XhPa
            spread = MIN_SPREAD
        elif spread > MAX_SPREAD: # more than XhPa
            spread = MAX_SPREAD

        utils.write_qc_config(config_file, "PRESSURE", "average", "{}".format(average), diagnostics=diagnostics)
        utils.write_qc_config(config_file, "PRESSURE", "spread", "{}".format(spread), diagnostics=diagnostics)

    return # identify_values

#************************************************************************
def pressure_offset(sealp, stnlp, times, config_file, plots=False, diagnostics=False):
    """
    Flag locations where difference between station and sea-level pressure
    falls outside of bounds

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param str configfile: string for configuration file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(sealp.data.shape[0])])

    difference = sealp.data - stnlp.data

    if len(difference.compressed()) >= utils.DATA_COUNT_THRESHOLD:

        try:
            average = float(utils.read_qc_config(config_file, "PRESSURE", "average"))
            spread = float(utils.read_qc_config(config_file, "PRESSURE", "spread"))
        except KeyError:
            print("Information missing in config file")
            average = utils.average(difference)
            spread = utils.spread(difference)
            if spread < MIN_SPREAD: # less than XhPa
                spread = MIN_SPREAD
            elif spread > MAX_SPREAD: # more than XhPa
                spread = MAX_SPREAD

            utils.write_qc_config(config_file, "PRESSURE", "average", "{}".format(average), diagnostics=diagnostics)
            utils.write_qc_config(config_file, "PRESSURE", "spread", "{}".format(spread), diagnostics=diagnostics)

        if np.abs(np.ma.mean(difference) - np.ma.median(difference)) > THRESHOLD*spread:
            if diagnostics:
                print("Large difference between mean and median")
                print("Likely to have two populations of roughly equal size")
                print("Test won't work")
            pass
        else:
            high, = np.ma.where(difference > (average + (THRESHOLD*spread)))
            low, = np.ma.where(difference < (average - (THRESHOLD*spread)))

            # diagnostic plots
            if plots:
                bins = np.arange(np.round(difference.min())-1, np.round(difference.max())+1, 0.1)
                import matplotlib.pyplot as plt
                plt.clf()
                plt.hist(difference.compressed(), bins=bins)
                plt.axvline(x=(average + (THRESHOLD*spread)), ls="--", c="r")
                plt.axvline(x=(average - (THRESHOLD*spread)), ls="--", c="r")
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

    if diagnostics:

        print("Pressure {}".format(stnlp.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # pressure_offset

#*********************************************
def calc_slp(stnlp, elevation, temperature):
    '''
    Suggestion from Scott Stevens to calculate the SLP from the StnLP
    Presumes 15C if no temperature available
    '''
    filled_temperature = np.ma.copy(temperature)

    # find locations where we could calculate the SLP, but temperatures are missing
    missing_Ts, = np.where(np.logical_and(filled_temperature.mask == True, stnlp.mask == False))
    if len(missing_Ts) > 0:
        filled_temperature[missing_Ts] = 15.0

    factor = (1. - ((0.0065*elevation) / ((filled_temperature+273.15) + (0.0065*elevation))))

    sealp = stnlp * (factor ** -5.257)    

    return sealp # calc_slp

#************************************************************************
def pressure_theory(sealp, stnlp, temperature, times, elevation, plots=False, diagnostics=False):
    """
    Flag locations where difference between recorded and calculated sea-level pressure 
    falls outside of bounds

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param MetVar temperature: temperature object
    :param array times: datetime array
    :param float elevation: station elevation (m)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(sealp.data.shape[0])])

    theoretical_value = calc_slp(stnlp.data, elevation, temperature.data)

    difference = sealp.data - theoretical_value

    bad_locs, = np.ma.where(np.ma.abs(difference) > THEORY_THRESHOLD)

    # diagnostic plots
    if plots:
        bins = np.arange(np.round(np.ma.min(difference))-1, np.round(np.ma.max(difference))+1, 0.1)
        import matplotlib.pyplot as plt
        plt.clf()
        plt.hist(difference.compressed(), bins=bins)
        plt.axvline(x=THEORY_THRESHOLD, ls="--", c="r")
        plt.axvline(x=-THEORY_THRESHOLD, ls="--", c="r")
        plt.xlim([bins[0] - 1, bins[-1] + 1])
        plt.ylabel("Observations")
        plt.xlabel("Difference (hPa)")
        plt.show()


    if len(bad_locs) != 0:
        flags[bad_locs] = "p"
        if diagnostics:
            print("Pressure".format(stnlp.name))
            print("   Number of mismatches between recorded and theoretical SLPs {}".format(len(bad_locs)))
        if plots:
            for bad in bad_locs:
                plot_pressure(sealp, stnlp, times, bad)

    # flag both as not sure immediately where the issue lies
    stnlp.flags = utils.insert_flags(stnlp.flags, flags)
    sealp.flags = utils.insert_flags(sealp.flags, flags)

    if diagnostics:

        print("Pressure {}".format(stnlp.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # pressure_theory

#************************************************************************
def pcc(station, config_file, full=False, plots=False, diagnostics=False):
    """
    Extract the variables and pass to the Pressure Cross Checks

    :param Station station: Station Object for the station
    :param str configfile: string for configuration file
    :param bool full: run a full update
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    sealp = getattr(station, "sea_level_pressure")
    stnlp = getattr(station, "station_level_pressure")

    if full:
        identify_values(sealp, stnlp, station.times, config_file, plots=plots, diagnostics=diagnostics)
    pressure_offset(sealp, stnlp, station.times, config_file, plots=plots, diagnostics=diagnostics)

    temperature = getattr(station, "temperature")
    pressure_theory(sealp, stnlp, temperature, station.times, station.elev, plots=plots, diagnostics=diagnostics)

    return # pcc

#************************************************************************
if __name__ == "__main__":

    print("pressure cross checks")
#************************************************************************
