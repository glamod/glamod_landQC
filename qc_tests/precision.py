"""
Precision Cross Checks
^^^^^^^^^^^^^^^^^^^^^^

Check paired metrics for timestamps where the precision of one is different
to the other.  For example, if dewpoint is to single degree, but temperatures
are to 0.1C, then this affects the humidity checks.

Paired metrics (Primary/Secondary):
  Temperature - Dewpoint temperature

"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************

#*********************************************
def plot_flags(primary: utils. Meteorological_Variable,
               secondary: utils.Meteorological_Variable,
               times: np.array, locations: np.array) -> None:
    '''
    Plot each month of observations and the flags

    :param MetVar primary: Meteorological variable object
    :param MetVar secondary: Meteorological variable object
    :param array times: datetime array
    :param array locations: the locations which have been flagged

    :returns:
    '''
    import matplotlib.pyplot as plt

    # simple plot
    plt.clf()
    plt.plot(times[locations], primary.data[locations], 'k-', marker=".",
             label=primary.name.capitalize())
    plt.plot(times[locations], secondary.data[locations], 'b-', marker=".",
             label=secondary.name.capitalize())
    plt.plot(times[locations], secondary.data[locations], 'r*', ms=10)

    plt.legend(loc="upper right")
    plt.ylabel(primary.units)
    plt.show()

    return # plot_flags


#************************************************************************
def precision_cross_check(station: utils.Station, primary: utils.Meteorological_Variable,
                          secondary: utils.Meteorological_Variable, plots: bool=False,
                          diagnostics: bool=False) -> None:
    """
    Flag locations where precision of secondary is different from primary

    :param Station station: Station Object for the station
    :param MetVar primary: primary meteorological variable object
    :param MetVar secondary: secondary meteorological variable object
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(primary.data.shape[0])])
  
    # work through on a monthly basis
    for year in np.unique(station.years):
        for month in range(1, 13):
            month_locs, = np.where(np.logical_and(station.years == year,
                                                  station.months == month))
            # skip to next month if:
            if month_locs.shape[0] == 0:
                # no data in any variable
                continue

            if (len(primary.data[month_locs].compressed()) < utils.DATA_COUNT_THRESHOLD) or\
                (len(secondary.data[month_locs].compressed()) < utils.DATA_COUNT_THRESHOLD):
                # no data in either of these two variables
                continue

            primary_precision = utils.reporting_accuracy(primary.data[month_locs])
            secondary_precision = utils.reporting_accuracy(secondary.data[month_locs])

            if primary_precision != secondary_precision:
                # flag secondary only
                locs, = np.nonzero(secondary.data[month_locs].mask == False)
                flags[month_locs[locs]] = "n"

                # diagnostic plots
                if plots:
                    plot_flags(primary, secondary, station.times, month_locs)
                if diagnostics:
                    print(f"{year} {month} : {primary_precision} {secondary_precision} : {len(locs)}")


    # only flag the secondary
    secondary.flags = utils.insert_flags(secondary.flags, flags)

    logger.info(f"Precision {secondary.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")
    if diagnostics:
        print(f"Precision {secondary.name}")
        print(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # precision_cross_check


#************************************************************************
def pcc(station: utils.Station, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Extract the variables and pass to the Precision Cross Check

    :param Station station: Station Object for the station
    :param str config_dict: dictionary for configuration settings (unused here)
    :param bool full: run a full update (unused here)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # Temperature/Dewpoint check
    temperatures = getattr(station, "temperature")
    dewpoints = getattr(station, "dew_point_temperature")

    precision_cross_check(station, temperatures, dewpoints, plots=plots, diagnostics=diagnostics)
    
    # other pairs will appear here

    return # pcc

#************************************************************************
if __name__ == "__main__":

    print("precision cross checks")
#************************************************************************
