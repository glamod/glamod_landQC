"""
Excess Variance Checks
^^^^^^^^^^^^^^^^^^^^^^

Checks for months with higher/lower variance than expected

"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
from scipy.stats import skew
import datetime as dt

import qc_utils as utils
#************************************************************************


SPREAD_THRESHOLD = 6.
MIN_VALUES = 30
DATA_COUNT_THRESHOLD = 120
#************************************************************************
def prepare_data(obs_var, station, month, diagnostics=False, winsorize=True):

    anomalies = np.ma.zeros(obs_var.data.shape[0])
    anomalies.mask = np.ones(anomalies.shape[0])
    normed_anomalies = np.ma.copy(anomalies)

    mlocs, = np.where(station.months == month)
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
                hour_data = utils.winsorize(hour_data, 50)

        if len(hour_data) >= DATA_COUNT_THRESHOLD:
            hourly_clims[hour] = np.ma.mean(hour_data)
            hourly_clims.mask[hour] = False

        # make anomalies - keeping the order
        anomalies[hlocs] = obs_var.data[hlocs] - hourly_clims[hour]

    # for the month, normalise anomalies by spread
    spread = utils.spread(anomalies[mlocs])
    if spread < 1.5:
        spread = 1.5

    normed_anomalies[mlocs] = anomalies[mlocs] / spread

    # calculate the variance for each year in this single month.
    all_years = np.unique(station.years)

    variances = np.ma.zeros(all_years.shape[0])
    variances.mask = np.ones(all_years.shape[0])
    for y, year in enumerate(all_years):

        ymlocs, = np.where(np.logical_and(station.months == month, station.years == year)) 
        this_year = normed_anomalies[ymlocs]

        # HadISD used M.A.D.
        if this_year.compressed().shape[0] > MIN_VALUES:
            variances[y] = utils.spread(this_year)

    return variances # prepare_data

#************************************************************************
def find_thresholds(obs_var, station, config_file, plots=False, diagnostics=False, winsorize=True):
    """
    Use distribution to identify threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    # get hourly climatology for each month
    for month in range(1, 13):

        variances = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        average_variance = utils.average(variances)
        variance_spread = utils.spread(variances)

        utils.write_qc_config(config_file, "VARIANCE-{}".format(obs_var.name), "{}-average".format(month), "{}".format(average_variance), diagnostics=diagnostics)
        utils.write_qc_config(config_file, "VARIANCE-{}".format(obs_var.name), "{}-spread".format(month), "{}".format(variance_spread), diagnostics=diagnostics)

    return # find_thresholds

#************************************************************************
def variance_check(obs_var, station, config_file, plots=False, diagnostics=False, winsorize=True):
    """
    Use distribution to identify threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    # get hourly climatology for each month
    for month in range(1, 13):

        variances = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        average_variance = float(utils.read_qc_config(config_file, "VARIANCE-{}".format(obs_var.name), "{}-average".format(month)))
        variance_spread = float(utils.read_qc_config(config_file, "VARIANCE-{}".format(obs_var.name), "{}-spread".format(month)))


        bad_years, = np.where(np.abs(variances - average_variance) / variance_spread > SPREAD_THRESHOLD)

        all_years = np.unique(station.years)
        for year in bad_years:
            
            ym_locs, = np.where(np.logical_and(station.months == month, station.years == all_years[year])) 
            flags[ym_locs] = "V"

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Variance {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))


        # TODO - SLP/STNLP storm check

    return # variance_check

#************************************************************************
def evc(station, var_list, config_file, logfile="", plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Excess Variance Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param str logfile: string for log file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        find_thresholds(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)
        variance_check(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)


    return # evc

#************************************************************************
if __name__ == "__main__":
    
    print("checking excess variance")
#************************************************************************
