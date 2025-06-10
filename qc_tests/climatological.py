"""
Climatological Outlier Check
============================

Check for observations which fall outside the climatologically expected range.
A low pass filter reduces the effect of long-term changes.

"""
#************************************************************************
import numpy as np
from scipy.stats import skew
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************

FREQUENCY_THRESHOLD = 0.1
GAP_SIZE = 2
BIN_WIDTH = 0.5

#************************************************************************
def get_weights(monthly_anoms: np.ndarray, monthly_subset: np.ndarray,
                filter_subset: np.ndarray) -> float:
    '''
    Get the weights for the low pass filter.

    :param array monthly_anoms: monthly anomalies
    :param array monthly_subset: which values to take
    :param array filter_subset: which values to take
    :returns:
        weights - float
    '''

    filterweights = np.array([1., 2., 3., 2., 1.])

    if np.sum(filterweights[filter_subset] * np.ceil(monthly_anoms[monthly_subset] - np.floor(monthly_anoms[monthly_subset]))) == 0:
        weights = 0
    else:
        weights = np.sum(filterweights[filter_subset] * monthly_anoms[monthly_subset]) / \
            np.sum(filterweights[filter_subset] * np.ceil(monthly_anoms[monthly_subset] - np.floor(monthly_anoms[monthly_subset])))

    return weights # get_weights

#************************************************************************
def low_pass_filter(normed_anomalies: np.ndarray, station: utils.Station,
                    monthly_anoms: np.ndarray, month: int) -> np.ndarray:
    '''
    Run the low pass filter - get suitable ranges, get weights, and apply

    :param array normed_anomalies: input normalised anomalies
    :param Station station: station object
    :param array monthly_anoms: year average anomalies for calendar month
    :param int month: month being processed

    :returns: normed_anomalies - with weights applied
    '''

    all_years = np.unique(station.years)
    for year in range(all_years.shape[0]):

        if year == 0:
            monthly_range = np.arange(0, 3)[:all_years.shape[0]]
            filter_range = np.arange(2, 5)[:all_years.shape[0]]
        elif year == 1:
            monthly_range = np.arange(0, 4)[:all_years.shape[0]]
            filter_range = np.arange(1, 5)[:all_years.shape[0]]
        elif year == all_years.shape[0] - 2:
            monthly_range = np.arange(-4, 0)[-all_years.shape[0]:]
            filter_range = np.arange(0, 4)[-all_years.shape[0]:]
        elif year == all_years.shape[0] - 1:
            monthly_range = np.arange(-3, 0)[-all_years.shape[0]:]
            filter_range = np.arange(0, 3)[-all_years.shape[0]:]
        else:
            monthly_range = np.arange(year-2, year+3)
            filter_range = np.arange(5)

        if np.ma.sum(np.ma.abs(monthly_anoms[monthly_range])) != 0:

            weights = get_weights(monthly_anoms, monthly_range, filter_range)

            ymlocs, = np.where(np.logical_and(station.months == month, station.years == year))
            normed_anomalies[ymlocs] = normed_anomalies[ymlocs] - weights

    return normed_anomalies # low_pass_filter

#************************************************************************
def prepare_data(obs_var: utils.Meteorological_Variable, station: utils.Station,
                 month: int, diagnostics: bool = False, winsorize: bool = True) -> np.ma.array:
    """
    Prepare the data for the climatological check.  Makes anomalies and applies low-pass filter

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: which month to run on
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    anomalies = np.ma.zeros(obs_var.data.shape[0])
    anomalies.mask = np.ones(anomalies.shape[0])
    normed_anomalies = np.ma.copy(anomalies)

    mlocs, = np.where(station.months == month)

    nyears = len(np.unique(station.years[mlocs]))

    # need to have some data and in at least 5 years!
    if len(mlocs) >= utils.DATA_COUNT_THRESHOLD and nyears >= 5:

        anomalies.mask[mlocs] = False
        normed_anomalies.mask[mlocs] = False

        hourly_clims = np.ma.zeros(24)
        hourly_clims.mask = np.ones(24)
        for hour in range(24):

            # calculate climatology
            hlocs, = np.where(np.logical_and(station.months == month, station.hours == hour))

            hour_data = obs_var.data[hlocs]

            if winsorize:
                if len(hour_data.compressed()) > 10:
                    hour_data = utils.winsorize(hour_data, 5)

            if len(hour_data) >= utils.DATA_COUNT_THRESHOLD:
                hourly_clims[hour] = np.ma.mean(hour_data)
                hourly_clims.mask[hour] = False

            # make anomalies - keeping the order
            anomalies[hlocs] = obs_var.data[hlocs] - hourly_clims[hour]

        # if insufficient data at each hour, then no anomalies calculated
        if len(anomalies[mlocs].compressed()) >= utils.DATA_COUNT_THRESHOLD:

            # for the month, normalise anomalies by spread
            spread = utils.spread(anomalies[mlocs])
            if spread < 1.5:
                spread = 1.5

            normed_anomalies[mlocs] = anomalies[mlocs] / spread

            # apply low pass filter derived from monthly values
            all_years = np.unique(station.years)
            monthly_anoms = np.ma.zeros(all_years.shape[0])
            for y, year in enumerate(all_years):

                ylocs, = np.where(station.years == year)
                year_data = obs_var.data[ylocs]

                monthly_anoms[y] = utils.average(year_data)

            lp_filtered_anomalies = low_pass_filter(normed_anomalies, station, monthly_anoms, month)

            return lp_filtered_anomalies # prepare_data

        else:
            return anomalies # prepare_data
    else:
        return anomalies # prepare_data

#************************************************************************
def find_month_thresholds(obs_var: utils.Meteorological_Variable, station: utils.Station,
                          config_dict: dict, plots: bool = False, diagnostics: bool = False,
                          winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    # get hourly climatology for each month
    for month in range(1, 13):

        normalised_anomalies = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        if len(normalised_anomalies.compressed()) >= utils.DATA_COUNT_THRESHOLD:

            bins = utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name)
            bincentres = bins[1:] - (BIN_WIDTH/2)
            hist, bin_edges = np.histogram(normalised_anomalies.compressed(), bins)

            # using skew-gaussian, and inflating spread to account for that in the fit
            # NOTE: for some of the recent exceptional extremes, skew gaussian
            #       may still not be sufficient, and need to think perhaps of an
            #       alternative distribution
            gaussian_fit = utils.fit_gaussian(bincentres, hist, max(hist),
                                              mu=np.ma.median(normalised_anomalies),
                                              sig=1.5*utils.spread(normalised_anomalies),
                                              skew=skew(normalised_anomalies.compressed())
            )

            if len(gaussian_fit) == 3:
                fitted_curve = utils.gaussian(bincentres, gaussian_fit)
            elif len(gaussian_fit) == 4:
                fitted_curve = utils.skew_gaussian(bincentres, gaussian_fit)

            # diagnostic plots
            if plots:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.step(bins[1:], hist, color='k', where="pre")
                plt.yscale("log")

                plt.ylabel("Number of Observations")
                plt.xlabel(f"Scaled {obs_var.name.capitalize()}")
                plt.title(f"{station.id} - month {month}")

                plt.plot(bincentres, fitted_curve)
                plt.ylim([0.1, max(hist)*2])

            # use bins and curve to find points where curve is < FREQUENCY_THRESHOLD
            #  round up or down to be fully encompassing
            try:
                lower_threshold = bins[:-1][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres < 0))[0]][-1]
            except:
                lower_threshold = bins[0]
            try:
                upper_threshold = bins[1:][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres > 0))[0]][0]
            except:
                upper_threshold = bins[-1]

            if plots:
                plt.axvline(upper_threshold, c="r")
                plt.axvline(lower_threshold, c="r")
                plt.show()

            # Store values
            # add uthresh first, then lthresh
            try:
                config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"] = upper_threshold
            except KeyError:
                CD_uthresh = {f"{month}-uthresh" : upper_threshold}
                config_dict[f"CLIMATOLOGICAL-{obs_var.name}"] = CD_uthresh

            config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"] = lower_threshold

    return # find_month_thresholds

#************************************************************************
def monthly_clim(obs_var: utils.Meteorological_Variable, station: utils.Station, config_dict: dict,
                 logfile: str = "", plots: bool = False, diagnostics: bool = False, winsorize: bool = True):
    """
    Run through the variables and pass to the Distributional Gap Checks

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str configfile: dictionary for configuration settings
    :param str logfile: string for log file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    for month in range(1, 13):

        month_locs, = np.where(station.months == month)

        # note these are for the whole record, just this month is unmasked
        normalised_anomalies = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        if len(normalised_anomalies.compressed()) >= utils.DATA_COUNT_THRESHOLD:

            bins = utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name)
            # bincentres = bins[1:] - (BIN_WIDTH/2)
            hist, bin_edges = np.histogram(normalised_anomalies.compressed(), bins)

            try:
                upper_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"])
                lower_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"])
            except KeyError:
                print("Information missing in config file")
                find_month_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
                upper_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-uthresh"])
                lower_threshold = float(config_dict[f"CLIMATOLOGICAL-{obs_var.name}"][f"{month}-lthresh"])

            # now to find the gaps
            uppercount = len(np.where(normalised_anomalies > upper_threshold)[0])
            lowercount = len(np.where(normalised_anomalies < lower_threshold)[0])

            if uppercount > 0:
                gap_start = utils.find_gap(hist, bins, upper_threshold, GAP_SIZE)

                if gap_start != 0:
                    bad_locs, = np.ma.where(normalised_anomalies > gap_start) # all years for one month

                    # normalised_anomalies are for the whole record, just this month is unmasked
                    flags[bad_locs] = "C"

            if lowercount > 0:
                gap_start = utils.find_gap(hist, bins, lower_threshold, GAP_SIZE, upwards=False)

                if gap_start != 0:
                    bad_locs, = np.ma.where(normalised_anomalies < gap_start) # all years for one month

                    flags[bad_locs] = "C"

            # diagnostic plots
            if plots:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.step(bins[1:], hist, color='k', where="pre")
                plt.yscale("log")

                plt.ylabel("Number of Observations")
                plt.xlabel(f"Scaled {obs_var.name.capitalize()}")
                plt.title(f"{station.id} - month {month}")

                plt.ylim([0.1, max(hist)*2])
                plt.axvline(upper_threshold, c="r")
                plt.axvline(lower_threshold, c="r")

                bad_locs, = np.where(flags[month_locs] == "C")
                bad_hist, dummy = np.histogram(normalised_anomalies[month_locs][bad_locs], bins)
                plt.step(bins[1:], bad_hist, color='r', where="pre")

                plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Climatological {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # monthly_clim


#************************************************************************
def coc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Climatological Outlier Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # no point plotting twice!
        cplots = plots
        if plots and full:
            cplots = False

        if full:
            find_month_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        monthly_clim(obs_var, station, config_dict, plots=cplots, diagnostics=diagnostics)

    return # coc

