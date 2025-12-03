"""
Frequent Value Check
====================

Check for observation values which occur much more frequently than expected.
"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import utils
import qc_tests.qc_utils as qc_utils
#************************************************************************

ROLLING = 7  # looking for peaks in centre of $ROLLING bins
BIN_WIDTH = 1.0
RATIO = 0.5

def plot_frequent_values(bins, hist, suspect,
                         xlabel, title) -> None:  #  pragma: no cover
    """
    Plot the histogram, with suspect values identified
    """

    import matplotlib.pyplot as plt

    plt.step(bins[1:], hist, color='k', where="pre")
    plt.yscale("log")

    plt.ylabel("Number of Observations")
    plt.xlabel(xlabel)
    plt.title(title)

    bad_hist = np.copy(hist)
    for b, _ in enumerate(bad_hist):
        if bins[b] not in suspect:
            bad_hist[b] = 0

    plt.step(bins[1:], bad_hist, color='r', where="pre")
    plt.show()



def get_histogram(indata: np.ma.MaskedArray,
                  name: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Get the histogram for the data, and the bins of the correct size

    Parameters
    ----------
    indata : np.ma.MaskedArray
        The data for that calendar month
    name : str
        The name of the variable for further checks on the bins

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        Bin width, and arrays of the histogram and associated bins
    """

    # adjust bin widths according to reporting accuracy
    resolution = qc_utils.reporting_accuracy(indata)

    # and check further dependent on the MetVar
    if resolution <= 0.5:
        width = 0.5
    else:
        width = 1.0
    bins = qc_utils.create_bins(indata.compressed(), width, name)

    hist, _ = np.histogram(indata.compressed(), bins)

    assert len(bins) == len(hist) + 1

    return (width, hist, bins)


def scan_histogram(hist: np.ndarray, bins: np.ndarray) -> list[int]:
    """Scan through the histogram to pick out suspect values

    Parameters
    ----------
    hist : np.ndarray
        histogram values
    bins : np.ndarray
        histogram bins

    Returns
    -------
    list[int]
        list of indices of values which are suspiciously common
    """
    assert len(bins) == len(hist) + 1

    # Scan through the histogram
    #   check if a bin is the maximum of a local area ("ROLLING")
    suspect = []
    for b, bar in enumerate(hist):

        if (b >= (ROLLING//2)) and (b < (len(hist) - ROLLING//2)):
            target_bins = hist[b-(ROLLING//2) : b + (ROLLING//2) + 1]

            # if sufficient obs, maximum and contains > 50% (can be all) of the data
            if bar < utils.DATA_COUNT_THRESHOLD:
                # if insufficient obs, skip on
                continue
            if bar == target_bins.max():
                # if the maximum
                if (bar/target_bins.sum()) > RATIO:
                    # and more than 50% of the data (can be all of it)
                    suspect += [bins[b]]

    return suspect

#************************************************************************
def identify_values(obs_var: utils.MeteorologicalVariable,
                    station: utils.Station, config_dict: dict,
                    plots: bool = False, diagnostics: bool = False) -> None:
    """
    Use distribution to identify frequent values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO - do we want to go down the road of allowing resolution (and hence test)
    #           to vary over the p-o-r?  I.e. 1C in early, to 0.5C to 0.1C in different decades

    for month in range(1, 13):
        # get data for calendar month
        locs, = np.nonzero(station.months == month)
        month_data = obs_var.data[locs]

        if len(month_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
            # insufficient data, so write out empty config and move on
            try:
                config_dict[f"FREQUENT-{obs_var.name}"][f"{month}-width"] = -1
            except KeyError:
                CD_width = {f"{month}-width" : -1}
                config_dict[f"FREQUENT-{obs_var.name}"] = CD_width
            config_dict[f"FREQUENT-{obs_var.name}"][f"{month}"] = []
            continue

        # get the histogram and the bins
        width, hist, bins = get_histogram(month_data, obs_var.name)

        # store bin width
        try:
            config_dict[f"FREQUENT-{obs_var.name}"][f"{month}-width"] = width
        except KeyError:
            CD_width = {f"{month}-width" : f"{width}"}
            config_dict[f"FREQUENT-{obs_var.name}"] = CD_width

        # TODO: Check that there are enough non-unique bins

        suspect = scan_histogram(hist, bins)

        # diagnostic plots
        if plots:
            plot_frequent_values(bins, hist, suspect,
                                 obs_var.name.capitalize(),
                                 f"{station.id} - month {month}")

        # write out the thresholds...
        config_dict[f"FREQUENT-{obs_var.name}"][f"{month}"] = suspect

    return # identify_values


#************************************************************************
def frequent_values(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                    config_dict: dict, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Use config file to read frequent values.  Check each month to see if appear.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary storing critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    all_years = np.unique(station.years)

    # work through each month, and then year
    for month in range(1, 13):

        # read in bin-width and suspect bins for this month
        try:
            width = float(config_dict[f"FREQUENT-{obs_var.name}"][f"{month}-width"])
            suspect_bins = config_dict[f"FREQUENT-{obs_var.name}"][f"{month}"]
        except KeyError:
            identify_values(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
            width = float(config_dict[f"FREQUENT-{obs_var.name}"][f"{month}-width"])
            suspect_bins = config_dict[f"FREQUENT-{obs_var.name}"][f"{month}"]

        # skip on if nothing to find
        if len(suspect_bins) == 0:
            continue

        # work through each year
        for year in all_years:
            locs, = np.where(np.logical_and(station.months == month, station.years == year))

            month_data = obs_var.data[locs]

            # skip if no data
            if np.ma.count(month_data) == 0:
                continue

            month_flags = np.array(["" for i in range(month_data.shape[0])])

            # use stored bin widths
            bins = qc_utils.create_bins(month_data, width, obs_var.name)
            hist, _ = np.histogram(month_data, bins)

            # Re-scan through the histogram
            #   check if a bin is the maximum of a local area in this month
            suspect_monthly = scan_histogram(hist, bins)

            for sm_bin in suspect_monthly:
                if sm_bin in suspect_bins:
                # find observations (month & year) to flag!
                    flag_locs = np.where(np.logical_and(month_data >= sm_bin,
                                                        month_data < sm_bin+width))
                    month_flags[flag_locs] = "F"

            # copy flags for all years into main array
            flags[locs] = month_flags

            # diagnostic plots
            if plots:
                # pull out the
                bad_hist = np.copy(hist)
                for b, _ in enumerate(bad_hist):
                    if bins[b] not in suspect_monthly:
                        bad_hist[b] = 0

                plot_frequent_values(bins, hist, bad_hist,
                                    obs_var.name.capitalize(),
                                    f"{station.id} - {year}/{month}")


    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Frequent Values {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {np.count_nonzero(flags != '')}")

    return # frequent_values

#************************************************************************
def fvc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Frequent Value Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary containing configuration info
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # no point plotting twice!
        fplots = plots
        if plots and full:
            fplots = False

        # Frequent Values
        if full:
            identify_values(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)

        frequent_values(obs_var, station, config_dict, plots=fplots, diagnostics=diagnostics)


    return # fvc

