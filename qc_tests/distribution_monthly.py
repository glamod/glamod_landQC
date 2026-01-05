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

VALID_MONTHS = 5
MIN_OBS = 28*2 # two obs per day for the month (minimum subdaily)
SPREAD_LIMIT = 2 # two IQR/MAD/STD
MONTHlY_BIN_WIDTH = 0.25
LARGE_LIMIT = 5

def plot_monthly_distribution(indata: np.ndarray, bins: np.ndarray,
                              bad: np.ndarray, xlabel: str,
                              title: str) -> None:  # pragma: no cover
    """Plot distribution and flagged values"""

    import matplotlib.pyplot as plt
    hist, _ = np.histogram(indata, bins)
    plt.step(bins[1:], hist, color='k', where="pre")
    if len(bad) > 0:
        bad_hist, _ = np.histogram(indata[bad], bins)
        plt.step(bins[1:], bad_hist, color='r', where="pre")

    plt.ylabel("Number of Months")
    plt.xlabel(xlabel)
    plt.title(title)

    plt.show()


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
    all_years = np.unique(station.years)  # returns sorted values

    month_averages = np.ma.zeros(all_years.shape[0])
    month_averages.mask = np.ones(month_averages.shape[0])
    # spin through each year to get average for the calendar month selected

    for y, year in enumerate(all_years):
        locs, = np.where(np.logical_and(station.months == month, station.years == year))

        month_data = obs_var.data[locs]

        if np.ma.count(month_data) > MIN_OBS:

            month_averages[y] = np.ma.mean(month_data)

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

        bins = qc_utils.create_bins(standardised_months, MONTHlY_BIN_WIDTH, obs_var.name)

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
            plot_monthly_distribution(standardised_months, bins, bad,
                                      obs_var.name.capitalize(),
                                      f"{station.id} - month {month}")

    # append flags to object
    obs_var.store_flags(utils.insert_flags(obs_var.flags, flags))

    logger.info(f"Distribution (monthly) {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    # monthly_gap


#************************************************************************
def dgc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False,
        plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the monthly Distributional Gap Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration information
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # monthly gap
        if full:
            find_monthly_scaling(obs_var, station, config_dict, diagnostics=diagnostics)
        monthly_gap(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)

    # dgc

