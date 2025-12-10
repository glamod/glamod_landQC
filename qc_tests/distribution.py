"""
Distributional Gap Checks
=========================

1. Check distribution of monthly values and look for assymmetry
2. Check distribution of all observations and look for distinct populations
"""
#************************************************************************
import copy
import numpy as np
from scipy.stats import skew
import logging
logger = logging.getLogger(__name__)

import qc_tests.qc_utils as qc_utils
import utils
#************************************************************************
STORM_THRESHOLD = 5

VALID_MONTHS = 5
MIN_OBS = 28*2 # two obs per day for the month (minimum subdaily)
SPREAD_LIMIT = 2 # two IQR/MAD/STD
BIN_WIDTH = 0.25
LARGE_LIMIT = 5
GAP_SIZE = 2
FREQUENCY_THRESHOLD = 0.1
#************************************************************************
def prepare_monthly_data(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                         month: int, diagnostics: bool = False) -> np.ma.MaskedArray:
    """
    Extract monthly data and make average.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param bool diagnostics: turn on diagnostic output

    :returns: np.ma.MaskedArray
    """

    all_years = np.unique(station.years)

    month_averages = []
    # spin through each year to get average for the calendar month selected
    for year in all_years:
        locs, = np.where(np.logical_and(station.months == month, station.years == year))

        month_data = obs_var.data[locs]

        if np.ma.count(month_data) > MIN_OBS:

            month_averages += [np.ma.mean(month_data)]

    month_averages = np.ma.array(month_averages)

    return month_averages # prepare_monthly_data


#************************************************************************
def find_monthly_scaling(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                         config_dict: dict, diagnostics: bool = False) -> None:
    """
    Find scaling parameters for monthly values and store in config file

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool diagnostics: turn on diagnostic output
    """

    # all_years = np.unique(station.years)

    for month in range(1, 13):

        month_averages = prepare_monthly_data(obs_var, station, month, diagnostics=diagnostics)

        if len(month_averages.compressed()) >= VALID_MONTHS:

            # have months, now to standardise
            climatology = qc_utils.average(month_averages) # mean
            spread = qc_utils.spread(month_averages) # IQR currently
            if spread < SPREAD_LIMIT:
                spread = SPREAD_LIMIT

            # write out the scaling...
            try:
                config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-clim"] = climatology
            except KeyError:
                CD_clim = {f"{month}-clim": climatology}
                config_dict[f"MDISTRIBUTION-{obs_var.name}"] = CD_clim
            config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-spread"] = spread

        else:
            try:
                config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-clim"] = utils.MDI
            except KeyError:
                CD_clim = {f"{month}-clim": utils.MDI}
                config_dict[f"MDISTRIBUTION-{obs_var.name}"] = CD_clim
            config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-spread"] = utils.MDI

    # find_monthly_scaling


#************************************************************************
def monthly_gap(obs_var: utils.MeteorologicalVariable, station: utils.Station, config_dict: dict,
                plots: bool = False, diagnostics: bool = False) -> None:
    """
    Use distribution to identify assymetries.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])
    all_years = np.unique(station.years)

    for month in range(1, 13):

        month_averages = prepare_monthly_data(obs_var, station, month, diagnostics=diagnostics)

        # read in the scaling
        try:
            climatology = float(config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-clim"])
            spread = float(config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-spread"])
        except KeyError:
            find_monthly_scaling(obs_var, station, config_dict, diagnostics=diagnostics)
            climatology = float(config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-clim"])
            spread = float(config_dict[f"MDISTRIBUTION-{obs_var.name}"][f"{month}-spread"])


        if climatology == utils.MDI and spread == utils.MDI:
            # these weren't calculable, move on
            continue

        standardised_months = (month_averages - climatology) / spread

        bins = qc_utils.create_bins(standardised_months, BIN_WIDTH, obs_var.name)
        hist, bin_edges = np.histogram(standardised_months, bins)

        # flag months with very large offsets
        bad, = np.where(np.abs(standardised_months) >= LARGE_LIMIT)
        # now follow flag locations back up through the process
        for bad_month_id in bad:
            # year ID for this set of calendar months
            locs, = np.where(np.logical_and(station.months == month, station.years == all_years[bad_month_id]))
            flags[locs] = "D"

        # walk distribution from centre to find assymetry
        sort_order = standardised_months.argsort()
        mid_point = len(standardised_months) // 2
        good = True
        step = 1
        bad = []
        while good:

            if standardised_months[sort_order][mid_point - step] != standardised_months[sort_order][mid_point + step]:

                suspect_months = [np.abs(standardised_months[sort_order][mid_point - step]), \
                                      np.abs(standardised_months[sort_order][mid_point + step])]

                if min(suspect_months) != 0:
                    # not all clustered at origin

                    if max(suspect_months)/min(suspect_months) >= 2. and min(suspect_months) >= 1.5:
                        # at least 1.5x spread from centre and difference of two in location (longer tail)
                        # flag everything further from this bin for that tail
                        if suspect_months[0] == max(suspect_months):
                            # LHS has issue (remember that have removed the sign)
                            bad = sort_order[:mid_point - (step-1)] # need -1 given array indexing standards
                        elif suspect_months[1] == max(suspect_months):
                            # RHS has issue
                            bad = sort_order[mid_point + step:]
                        good = False

            step += 1
            if (mid_point - step) < 0 or (mid_point + step) == standardised_months.shape[0]:
                # reached end
                break

        # now follow flag locations back up through the process
        for bad_month_id in bad:
            # year ID for this set of calendar months
            locs, = np.where(np.logical_and(station.months == month, station.years == all_years[bad_month_id]))
            flags[locs] = "D"

        if plots:
            import matplotlib.pyplot as plt

            plt.step(bins[1:], hist, color='k', where="pre")
            if len(bad) > 0:
                bad_hist, dummy = np.histogram(standardised_months[bad], bins)
                plt.step(bins[1:], bad_hist, color='r', where="pre")

            plt.ylabel("Number of Months")
            plt.xlabel(obs_var.name.capitalize())
            plt.title(f"{station.id} - month {month}")

            plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Distribution (monthly) {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # monthly_gap


#************************************************************************
def prepare_all_data(obs_var: utils.MeteorologicalVariable,
                     station: utils.Station, month: int,
                     config_dict: dict, full: bool = False,
                     diagnostics: bool = False) -> np.ma.MaskedArray:
    """
    Extract data for the month, make & store or read average and spread.
    Use to calculate normalised anomalies.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param str config_dict: configuration dictionary to store critical values
    :param bool diagnostics: turn on diagnostic output

    :returns: np.ma.MaskedArray
    """

    month_locs, = np.where(station.months == month)

    all_month_data = obs_var.data[month_locs]

    if full:

        if len(all_month_data.compressed()) >= utils.DATA_COUNT_THRESHOLD:
            # have data, now to standardise
            climatology = qc_utils.average(all_month_data) # mean
            spread = qc_utils.spread(all_month_data) # IQR currently
        else:
            climatology = utils.MDI
            spread = utils.MDI

        # write out the scaling...
        try:
            config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-clim"] = climatology
        except KeyError:
            CD_clim = {f"{month}-clim" : climatology}
            config_dict[f"ADISTRIBUTION-{obs_var.name}"] = CD_clim
        config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-spread"] = spread

    else:

        try:
            climatology = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-clim"])
            spread = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-spread"])
        except KeyError:

            if len(all_month_data.compressed()) >= utils.DATA_COUNT_THRESHOLD:
                # have data, now to standardise
                climatology = qc_utils.average(all_month_data) # mean
                spread = qc_utils.spread(all_month_data) # IQR currently
            else:
                climatology = utils.MDI
                spread = utils.MDI

            # write out the scaling...
            try:
                config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-clim"] = climatology
            except KeyError:
                CD_clim = {f"{month}-clim" : climatology}
                config_dict[f"ADISTRIBUTION-{obs_var.name}"] = CD_clim
            config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-spread"] = spread

    if climatology == utils.MDI and spread == utils.MDI:
        # these weren't calculable, move on
        return np.ma.array([utils.MDI])
    elif spread == 0:
        # all the same value
        return (all_month_data - climatology)  # prepare_all_data
    else:
        return (all_month_data - climatology)/spread  # prepare_all_data

#************************************************************************
def find_thresholds(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                    config_dict: dict, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Extract data for month and find thresholds in distribution and store.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param str config_dict: configuration file to store critical values
    :param bool diagnostics: turn on diagnostic output
    """

    for month in range(1, 13):

        normalised_anomalies = prepare_all_data(obs_var, station, month, config_dict, full=True, diagnostics=diagnostics)

        if len(normalised_anomalies.compressed()) == 1 and normalised_anomalies[0] == utils.MDI:
            # scaling not possible for this month
            # add uthresh first, then lthresh
            try:
                config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"] = utils.MDI
            except KeyError:
                CD_uthresh = {f"{month}-uthresh" : utils.MDI}
                config_dict[f"ADISTRIBUTION-{obs_var.name}"] = CD_uthresh
            config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"] = utils.MDI
            continue
        elif len(np.unique(normalised_anomalies)) == 1:
            # all the same value, so won't be able to fit a histogram
            # add uthresh first, then lthresh
            try:
                config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"] = utils.MDI
            except KeyError:
                CD_uthresh = {f"{month}-uthresh" : utils.MDI}
                config_dict[f"ADISTRIBUTION-{obs_var.name}"] = CD_uthresh
            config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"] = utils.MDI
            continue

        bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name, anomalies=True)
        bincentres = bins[1:] - (BIN_WIDTH/2)
        hist, bin_edges = np.histogram(normalised_anomalies, bins)

        gaussian_fit = qc_utils.fit_gaussian(bincentres, hist, 0.5*max(hist),
                                          mu=np.ma.median(normalised_anomalies),
                                          sig=1.5*qc_utils.spread(normalised_anomalies),
                                          skew=0.5*skew(normalised_anomalies.compressed()))

        fitted_curve = qc_utils.skew_gaussian(bincentres, gaussian_fit)

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Observations")
            plt.xlabel(f"Normalised {obs_var.name.capitalize()} anomalies")
            plt.title(f"{station.id} - month {month}")

            plt.plot(bincentres, fitted_curve)
            plt.ylim([0.1, max(hist)*2])

        # use bins and curve to find points where curve is < FREQUENCY_THRESHOLD
        try:
            lower_threshold = bincentres[np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres < bins[np.argmax(fitted_curve)]))[0]][-1]
        except:
            lower_threshold = bins[1]
        try:
            if len(np.unique(fitted_curve)) == 1:
                # just a line of zeros perhaps (found on AFA00409906 station_level_pressure 20190913)
                upper_threshold = bins[-1]
            else:
                upper_threshold = bincentres[np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bincentres > bins[np.argmax(fitted_curve)]))[0]][0]
        except:
            upper_threshold = bins[-1]

        if plots:
            plt.axvline(upper_threshold, c="r")
            plt.axvline(lower_threshold, c="r")
            plt.show()

        # add uthresh first, then lthresh
        try:
            config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"] = upper_threshold
        except KeyError:
            CD_uthresh = {f"{month}-uthresh" : upper_threshold}
            config_dict[f"ADISTRIBUTION-{obs_var.name}"] = CD_uthresh
        config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"] = lower_threshold

    # find_thresholds

#************************************************************************
def expand_around_storms(storms: np.ndarray, maximum: int, pad: int = 6) -> np.ndarray:
    """
    Pad storm signal by N=6 hours

    """
    for i in range(pad):

        if storms[0]-1 >= 0:
            storms = np.insert(storms, 0, storms[0]-1)
        if storms[-1]+1 < maximum:
            storms = np.append(storms, storms[-1]+1)

    return np.unique(storms) # expand_around_storms

#************************************************************************
def all_obs_gap(obs_var: utils.MeteorologicalVariable, station: utils.Station,
                config_dict: dict, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Extract data for month and find secondary populations in distribution.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    for month in range(1, 13):

        normalised_anomalies = prepare_all_data(obs_var, station, month, config_dict, full=False, diagnostics=diagnostics)

        if (len(normalised_anomalies.compressed()) == 1 and normalised_anomalies[0] == utils.MDI):
            # no data to work with for this month, move on.
            continue

        bins = qc_utils.create_bins(normalised_anomalies, BIN_WIDTH, obs_var.name, anomalies=True)
        hist, bin_edges = np.histogram(normalised_anomalies, bins)

        try:
            upper_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"])
            lower_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"])
        except KeyError:
            find_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
            upper_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-uthresh"])
            lower_threshold = float(config_dict[f"ADISTRIBUTION-{obs_var.name}"][f"{month}-lthresh"])


        if upper_threshold == utils.MDI and lower_threshold == utils.MDI:
            # these weren't able to be calculated, move on
            continue
        elif len(np.unique(normalised_anomalies)) == 1:
            # all the same value, so won't be able to fit a histogram
            continue

        # now to find the gaps
        uppercount = len(np.where(normalised_anomalies > upper_threshold)[0])
        lowercount = len(np.where(normalised_anomalies < lower_threshold)[0])

        month_locs, = np.where(station.months == month) # append should keep year order
        if uppercount > 0:
            gap_start = qc_utils.find_gap(hist, bins, upper_threshold, GAP_SIZE)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies > gap_start) # all years for one month

                month_flags = flags[month_locs]
                month_flags[bad_locs] = "d"
                flags[month_locs] = month_flags

        if lowercount > 0:
            gap_start = qc_utils.find_gap(hist, bins, lower_threshold, GAP_SIZE, upwards=False)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies < gap_start) # all years for one month

                month_flags = flags[month_locs]
                month_flags[bad_locs] = "d"

                # TODO - can this bit be refactored?
                # for pressure data, see if the flagged obs correspond with high winds
                # could be a storm signal
                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    wind_monthly_data = prepare_monthly_data(station.wind_speed, station, month)
                    pressure_monthly_data = prepare_monthly_data(obs_var, station, month)

                    if len(pressure_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
                            len(wind_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
                        # need sufficient data to work with for storm check to work, else can't tell
                        pass
                    else:

                        wind_monthly_average = qc_utils.average(wind_monthly_data)
                        wind_monthly_spread = qc_utils.spread(wind_monthly_data)

                        pressure_monthly_average = qc_utils.average(pressure_monthly_data)
                        pressure_monthly_spread = qc_utils.spread(pressure_monthly_data)

                        # already a single calendar month, so go through each year
                        all_years = np.unique(station.years)
                        for year in all_years:

                            # what's best - extract only when necessary but repeatedly if so, or always, but once
                            this_year_locs = np.where(station.years[month_locs] == year)

                            if "d" not in month_flags[this_year_locs]:
                                # skip if you get the chance
                                continue

                            wind_data = station.wind_speed.data[month_locs][this_year_locs]
                            pressure_data = obs_var.data[month_locs][this_year_locs]

                            storms, = np.ma.where(np.logical_and((((wind_data - wind_monthly_average)/wind_monthly_spread) > STORM_THRESHOLD), (((pressure_monthly_average - pressure_data)/pressure_monthly_spread) > STORM_THRESHOLD)))

                            # more than one entry - check if separate events
                            if len(storms) >= 2:
                                # find where separation more than the usual obs separation
                                storm_1diffs = np.ma.diff(storms)
                                separations, = np.where(storm_1diffs > np.ma.median(np.ma.diff(wind_data)))

                                if len(separations) != 0:
                                    # multiple storm signals
                                    storm_start = 0
                                    storm_finish = separations[0] + 1
                                    first_storm = expand_around_storms(storms[storm_start: storm_finish], len(wind_data))
                                    final_storm_locs = copy.deepcopy(first_storm)

                                    for j in range(len(separations)):
                                        # then do the rest in a loop

                                        if j+1 == len(separations):
                                            # final one
                                            this_storm = expand_around_storms(storms[separations[j]+1: ], len(wind_data))
                                        else:
                                            this_storm = expand_around_storms(storms[separations[j]+1: separations[j+1]+1], len(wind_data))

                                        final_storm_locs = np.append(final_storm_locs, this_storm)

                                else:
                                    # locations separated at same interval as data
                                    final_storm_locs = expand_around_storms(storms, len(wind_data))

                            # single entry
                            elif len(storms) != 0:
                                # expand around the storm signal (rather than
                                #  just unflagging what could be the peak and
                                #  leaving the entry/exit flagged)
                                final_storm_locs = expand_around_storms(storms, len(wind_data))

                            # unset the flags
                            if len(storms) > 0:
                                month_flags[this_year_locs][final_storm_locs] = ""

                # having checked for storms now store final flags
                flags[month_locs] = month_flags


        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt
            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Observations")
            plt.xlabel(obs_var.name.capitalize())
            plt.title(f"{station.id} - month {month}")

            plt.ylim([0.1, max(hist)*2])

            plt.axvline(upper_threshold, c="r")
            plt.axvline(lower_threshold, c="r")

            bad_locs, = np.where(flags[month_locs] == "d")
            bad_hist, dummy = np.histogram(normalised_anomalies[bad_locs], bins)
            plt.step(bins[1:], bad_hist, color='r', where="pre")

            plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)


    logger.info(f"Distribution (all) {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # all_obs_gap

#************************************************************************
def dgc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Distributional Gap Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration information
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # no point plotting twice!
        gplots = plots
        if plots and full:
            gplots = False

        # monthly gap
        if full:
            find_monthly_scaling(obs_var, station, config_dict, diagnostics=diagnostics)
        monthly_gap(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)

        # all observations gap
        if full:
            find_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        all_obs_gap(obs_var, station, config_dict, plots=gplots, diagnostics=diagnostics)


    # dgc

