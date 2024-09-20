"""
Pressure Cross Checks
^^^^^^^^^^^^^^^^^^^^^

Check for observations where difference between station and sea level pressure
falls outside of the expected range.
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************

THRESHOLD = 4 # min spread of 1hPa, so only outside +/-4hPa flagged.
THEORY_THRESHOLD = 15 # allows for small offset as well as intrinsic spread

MIN_SPREAD = 1.0
MAX_SPREAD = 5.0

#*********************************************
def plot_pressure_timeseries(sealp: utils.Meteorological_Variable,
                             stnlp: utils.Meteorological_Variable,
                             times: np.array, bad: int) -> None:
    '''
    Plot each observation of SLP or StnLP against surrounding data

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param int bad: the location of SLP or StnLP
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
    plt.plot(times[pad_start : pad_end], sealp.data[pad_start : pad_end], 'k-',
             marker=".", label=sealp.name.capitalize())
    plt.plot(times[pad_start : pad_end], stnlp.data[pad_start : pad_end], 'b-',
             marker=".", label=stnlp.name.capitalize())
    plt.plot(times[bad], sealp.data[bad], 'r*', ms=10)
    plt.plot(times[bad], stnlp.data[bad], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(sealp.units)

    plt.show()

    return # plot_pressure_timeseries


#*********************************************
def plot_pressure_distribution(difference: np.array, vmin: int=-1, vmax:int=1) -> None:
    '''
    Plot distribution and include the upper and lower thresholds

    :param array difference: values to form histogram from 
    :param int vmin: lower locations for vertical line
    :param int vmax: upper locations for vertical line
    '''
    import matplotlib.pyplot as plt

    bins = np.arange(np.round(difference.min())-1,
                     np.round(difference.max())+1, 0.1)

    plt.clf()
    plt.hist(difference.compressed(), bins=bins)
    plt.axvline(x=vmin, ls="--", c="r")
    plt.axvline(x=vmax, ls="--", c="r")
    plt.xlim([bins[0] - 1, bins[-1] + 1])
    plt.ylabel("Observations")
    plt.xlabel("Difference (hPa)")
    plt.show()

    return # plot_pressure_distribution


#************************************************************************
def identify_values(sealp: utils.Meteorological_Variable,
                    stnlp: utils.Meteorological_Variable,
                    config_dict: dict,
                    plots: bool=False, diagnostics: bool=False) -> None:
    """
    Find average and spread of differences

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param str config_dict: dictionary for configuration settings
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

        try:
            config_dict["PRESSURE"]["average"] = average
        except KeyError:
            # Make new entry for Pressure
            CD_average = {"average" : average}
            config_dict["PRESSURE"] = CD_average
            
        config_dict["PRESSURE"]["spread"] = spread

    return # identify_values


#************************************************************************
def pressure_offset(sealp: utils.Meteorological_Variable,
                    stnlp: utils.Meteorological_Variable,
                    times: np.array, config_dict: dict,
                    plots: bool=False, diagnostics: bool=False) -> None:

    """
    Flag locations where difference between station and sea-level pressure
    falls outside of bounds

    :param MetVar sealp: sea level pressure object
    :param MetVar stnlp: station level pressure object
    :param array times: datetime array
    :param str config_dict: dictionary for configuration settings
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(sealp.data.shape[0])])

    difference = sealp.data - stnlp.data

    if len(difference.compressed()) >= utils.DATA_COUNT_THRESHOLD:

        try:
            average = float(config_dict["PRESSURE"]["average"])
            spread = float(config_dict["PRESSURE"]["spread"])
        except KeyError:
            identify_values(sealp, stnlp, config_dict, plots=plots, diagnostics=diagnostics)
            average = float(config_dict["PRESSURE"]["average"])
            spread = float(config_dict["PRESSURE"]["spread"])
            

        if np.abs(np.ma.mean(difference) - np.ma.median(difference)) > THRESHOLD*spread:
            logger.warn("Large difference between mean and median")
            logger.warn("Likely to have two populations of roughly equal size")
            logger.warn("Test won't work")
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
                plot_pressure_distribution(difference, vmin=(average + (THRESHOLD*spread)),
                                           vmax=(average - (THRESHOLD*spread)))

            if len(high) != 0 or len(low) != 0:
                logger.info(f"Pressure {stnlp.name}")
                print(f"Pressure {stnlp.name}")
            if len(high) != 0:
                flags[high] = "p"
                logger.info(f"   Number of high differences {len(high)}")
                if diagnostics:
                    print(f"   Number of high differences {len(high)}")
                if plots:
                    for bad in high:
                        plot_pressure_timeseries(sealp, stnlp, times, bad)

            if len(low) != 0:
                flags[low] = "p"
                logger.info(f"   Number of low differences {len(low)}")
                if diagnostics:
                    print(f"   Number of low differences {len(low)}")
                if plots:
                    for bad in low:
                        plot_pressure_timeseries(sealp, stnlp, times, bad)

            # only flag the station level pressure
            stnlp.flags = utils.insert_flags(stnlp.flags, flags)

    logger.info(f"Pressure {stnlp.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")
    if diagnostics:
        print(f"Pressure {stnlp.name}")
        print(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # pressure_offset


#*********************************************
def calc_slp(stnlp: np.array, elevation: float, temperature: np.array) -> np.array:
    '''
    Suggestion from Scott Stevens to calculate the SLP from the StnLP
    Presumes 15C if no temperature available

    Formula from https://keisan.casio.com/keisan/image/Convertpressure.pdf

    :param array stnlp: station level pressure data
    :param float elevation: station elevation
    :param array temperature: temperature data

    :returns: np.array
    '''

    filled_temperature = np.ma.copy(temperature)

    # find locations where we could calculate the SLP, but temperatures are missing
    missing_Ts, = np.where(np.logical_and(filled_temperature.mask == True,
                                          stnlp.mask == False))
    if len(missing_Ts) > 0:
        filled_temperature[missing_Ts] = 15.0

    factor = (1. - ((0.0065*elevation) / ((filled_temperature+273.15) + (0.0065*elevation))))

    sealp = stnlp * (factor ** -5.257)    

    return sealp # calc_slp


#************************************************************************
def adjust_existing_flag_locs(var: utils.Meteorological_Variable,
                            flags: np.array) -> np.array:
    """
    There may be flags already set by previous part of test
    Find these locations, and adjust new flags to these aren't added again
    
    :param MetVar var: the variable object
    :param array flags: the flag array

    :returns: updated flag array
    """

    pre_exist = [i for i,item in enumerate(var.flags) if "p" in item]
    new_flags = flags[:]

    # remove flags if "p" already in the existing flag so as not to duplicate
    new_flags[pre_exist] = ""

    return new_flags # adjust_existing_flag_locs


#************************************************************************
def pressure_theory(sealp: utils.Meteorological_Variable,
                    stnlp: utils.Meteorological_Variable,
                    temperature: utils.Meteorological_Variable,
                    times: np.array, elevation: int,
                    plots: bool=False, diagnostics: bool=False) -> None:
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
        plot_pressure_distribution(difference, vmin=-THEORY_THRESHOLD,
                                   vmax=THEORY_THRESHOLD)

    if len(bad_locs) != 0:
        flags[bad_locs] = "p"
        logger.info(f"Pressure {stnlp.name}")
        logger.info(f"   Number of mismatches between recorded and theoretical SLPs {len(bad_locs)}")
        if diagnostics:
            print(f"Pressure {stnlp.name}")
            print(f"   Number of mismatches between recorded and theoretical SLPs {len(bad_locs)}")
        if plots:
            for bad in bad_locs:
                plot_pressure_timeseries(sealp, stnlp, times, bad)
                
    # flag both as not sure immediately where the issue lies
    stnlp.flags = utils.insert_flags(stnlp.flags, adjust_existing_flag_locs(stnlp, flags))
    sealp.flags = utils.insert_flags(sealp.flags, adjust_existing_flag_locs(sealp, flags))

    logger.info(f"Pressure {stnlp.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")
    if diagnostics:
        print(f"Pressure {stnlp.name}")
        print(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # pressure_theory


#************************************************************************
def pcc(station: utils.Station, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Extract the variables and pass to the Pressure Cross Checks

    :param Station station: Station Object for the station
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    sealp = getattr(station, "sea_level_pressure")
    stnlp = getattr(station, "station_level_pressure")

    if full:
        identify_values(sealp, stnlp, config_dict, plots=plots, diagnostics=diagnostics)
    pressure_offset(sealp, stnlp, station.times, config_dict, plots=plots, diagnostics=diagnostics)

    temperature = getattr(station, "temperature")
    if str(station.elev)[:4] in ["-999", "9999"]:
        # missing elevation, so can't run this check
        logger.warn(f"Station Elevation missing ({station.elev}m)")
        logger.warn("   Theoretical SLP/StnLP cross check not run.")
        print(f"Station Elevation missing ({station.elev}m)")
        print("   Theoretical SLP/StnLP cross check not run.")
    else:
        pressure_theory(sealp, stnlp, temperature, station.times, station.elev, plots=plots, diagnostics=diagnostics)

    return # pcc


#************************************************************************
if __name__ == "__main__":

    print("pressure cross checks")
#************************************************************************
