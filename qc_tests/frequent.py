"""
Frequent Value Check
^^^^^^^^^^^^^^^^^^^^

Check for observation values which occur much more frequently than expected.
"""
#************************************************************************
import numpy as np

import qc_utils as utils
#************************************************************************

ROLLING = 7
BIN_WIDTH = 1.0
RATIO = 0.5

#************************************************************************
def identify_values(obs_var, station, config_dict, plots=False, diagnostics=False):
    """
    Use distribution to identify frequent values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # TODO - do we want to go down the road of allowing resolution (and hence test)
    #           to vary over the p-o-r?  I.e. 1C in early, to 0.5C to 0.1C in different decades?

    # store bin width
    CD_width = {"width" : "{}".format(BIN_WIDTH)}
    config_dict["FREQUENT-{}".format(obs_var.name)] = CD_width

    for month in range(1, 13):

        locs, = np.where(station.months == month)

        month_data = obs_var.data[locs]

        if len(month_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
            # insufficient data, so write out empty config and move on
            config_dict["FREQUENT-{}".format(obs_var.name)]["{}".format(month)] = []
            continue

        # adjust bin widths according to reporting accuracy
        resolution = utils.reporting_accuracy(month_data)

        if resolution <= 0.5:
            bins = utils.create_bins(month_data, 0.5, obs_var.name) 
        else:
            bins = utils.create_bins(month_data, 1.0, obs_var.name) 

        hist, bin_edges = np.histogram(month_data, bins)

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt

            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Observations")
            plt.xlabel(obs_var.name.capitalize())
            plt.title("{} - month {}".format(station.id, month))

        # Scan through the histogram
        #   check if a bin is the maximum of a local area ("ROLLING")
        suspect = []
        for b, bar in enumerate(hist):
            if (b > ROLLING//2) and (b <= (len(hist) - ROLLING//2)):

                target_bins = hist[b-(ROLLING//2) : b + (ROLLING//2) + 1]

                # if sufficient obs, maximum and contains > 50%, but not all, of the data
                if bar >= utils.DATA_COUNT_THRESHOLD:
                    if bar == target_bins.max():
                        if (bar/target_bins.sum()) > RATIO:
                            suspect += [bins[b]]

        # diagnostic plots
        if plots:
            bad_hist = np.copy(hist)
            for b, bar in enumerate(bad_hist):
                if bins[b] not in suspect:
                    bad_hist[b] = 0

            plt.step(bins[1:], bad_hist, color='r', where="pre")
            plt.show()


        # write out the thresholds...
        config_dict["FREQUENT-{}".format(obs_var.name)]["{}".format(month)] = suspect

    return # identify_values


#************************************************************************
def frequent_values(obs_var, station, config_dict, plots=False, diagnostics=False):
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
            width = float(config_dict["FREQUENT-{}".format(obs_var.name)]["width"])
            suspect_bins = config_dict["FREQUENT-{}".format(obs_var.name)]["{}".format(month)]
        except KeyError:
            print("Information missing in config dictionary")
            identify_values(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
            width = float(config_dict["FREQUENT-{}".format(obs_var.name)]["width"])
            suspect_bins = config_dict["FREQUENT-{}".format(obs_var.name)]["{}".format(month)]
 
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

            # adjust bin widths according to reporting accuracy
            resolution = utils.reporting_accuracy(month_data)

            if resolution <= 0.5:
                bins = utils.create_bins(month_data, 0.5, obs_var.name) 
            else:
                bins = utils.create_bins(month_data, 1.0, obs_var.name) 
            hist, bin_edges = np.histogram(month_data, bins)

            # Scan through the histogram
            #   check if a bin is the maximum of a local area ("ROLLING")
            for b, bar in enumerate(hist):
                if (b > ROLLING//2) and (b <= (len(hist) - ROLLING//2)):

                    target_bins = hist[b-(ROLLING//2) : b + (ROLLING//2) + 1]

                    # if sufficient obs, maximum and contains > 50% of data
                    if bar >= utils.DATA_COUNT_THRESHOLD:
                        if bar == target_bins.max():
                            if (bar/target_bins.sum()) > RATIO:
                                # this bin meets all the criteria
                                if bins[b] in suspect_bins:
                                    # find observations (month & year) to flag!
                                    flag_locs = np.where(np.logical_and(month_data >= bins[b], month_data < bins[b+1]))
                                    month_flags[flag_locs] = "F"

            # copy flags for all years into main array
            flags[locs] = month_flags

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt

            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Observations")
            plt.xlabel(obs_var.name.capitalize())
            plt.title("{} - month {}".format(station.id, month))

            bad_hist = np.copy(hist)
            for b, bar in enumerate(bad_hist):
                if bins[b] not in suspect_bins:
                    bad_hist[b] = 0

            plt.step(bins[1:], bad_hist, color='r', where="pre")
            plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:

        print("Frequent Values {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # frequent_values

#************************************************************************
def fvc(station, var_list, config_dict, full=False, plots=False, diagnostics=False):
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

#************************************************************************
if __name__ == "__main__":

    print("checking frequent values")
#************************************************************************
