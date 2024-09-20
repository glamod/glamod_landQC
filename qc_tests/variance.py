"""
Excess Variance Checks
^^^^^^^^^^^^^^^^^^^^^^

Checks for months with higher/lower variance than expected

"""
#************************************************************************
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils
#************************************************************************
STORM_THRESHOLD = 4
MIN_VARIANCES = 10
SPREAD_THRESHOLD = 8.
MIN_VALUES = 30

#************************************************************************
def prepare_data(obs_var: utils.Meteorological_Variable, station: utils.Station, month:int, diagnostics: bool = False, winsorize: bool = True) -> np.array:
    """
    Calculate the monthly variances

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

        if len(hour_data.compressed()) >= utils.DATA_COUNT_THRESHOLD:
            hourly_clims[hour] = np.ma.mean(hour_data)
            hourly_clims.mask[hour] = False

        # make anomalies - keeping the order
        anomalies[hlocs] = obs_var.data[hlocs] - hourly_clims[hour]


    if len(anomalies[mlocs].compressed()) >= MIN_VARIANCES:
        # for the month, normalise anomalies by spread
        spread = utils.spread(anomalies[mlocs])
        if spread < 1.5:
            spread = 1.5
    else:
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
def find_thresholds(obs_var: utils.Meteorological_Variable, station: utils.Station, config_dict: dict,
                    plots: bool = False, diagnostics: bool = False, winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config dictionary.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration dictionary to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    # get hourly climatology for each month
    for month in range(1, 13):

        variances = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        if len(variances.compressed()) >= MIN_VARIANCES:
            average_variance = utils.average(variances)
            variance_spread = utils.spread(variances)
        else:
            average_variance = utils.MDI
            variance_spread = utils.MDI

        try:
            config_dict["VARIANCE-{}".format(obs_var.name)]["{}-average".format(month)] =  average_variance
        except KeyError:
            CD_average = {"{}-average".format(month) : average_variance}
            config_dict["VARIANCE-{}".format(obs_var.name)] = CD_average
            
        config_dict["VARIANCE-{}".format(obs_var.name)]["{}-spread".format(month)] = variance_spread

    return # find_thresholds

#************************************************************************
def variance_check(obs_var: utils.Meteorological_Variable, station: utils.Station, config_dict: dict, plots: bool = False, diagnostics: bool = False, winsorize: bool = True) -> None:
    """
    Use distribution to identify threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_dict: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    :param bool winsorize: apply winsorization at 5%/95%
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    # get hourly climatology for each month
    for month in range(1, 13):
        month_locs, = np.where(station.months == month)

        variances = prepare_data(obs_var, station, month, diagnostics=diagnostics, winsorize=winsorize)

        try:
            average_variance = float(config_dict["VARIANCE-{}".format(obs_var.name)]["{}-average".format(month)])
            variance_spread = float(config_dict["VARIANCE-{}".format(obs_var.name)]["{}-spread".format(month)])
        except KeyError:
            find_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
            average_variance = float(config_dict["VARIANCE-{}".format(obs_var.name)]["{}-average".format(month)])
            variance_spread = float(config_dict["VARIANCE-{}".format(obs_var.name)]["{}-spread".format(month)])

        if average_variance == utils.MDI and variance_spread == utils.MDI:
            # couldn't be calculated, move on
            continue

        bad_years, = np.where(np.abs(variances - average_variance) / variance_spread > SPREAD_THRESHOLD)

        # prepare wind and pressure data in case needed to check for storms
        if obs_var.name in ["station_level_pressure", "sea_level_pressure", "wind_speed"]:
            wind_monthly_data = station.wind_speed.data[month_locs]
            if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                pressure_monthly_data = obs_var.data[month_locs]
            else:
                pressure_monthly_data = station.sea_level_pressure.data[month_locs]

            if len(pressure_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
                    len(wind_monthly_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
                # need sufficient data to work with for storm check to work, else can't tell
                #    move on
                continue

            wind_average = utils.average(wind_monthly_data)
            wind_spread = utils.spread(wind_monthly_data)

            pressure_average = utils.average(pressure_monthly_data)
            pressure_spread = utils.spread(pressure_monthly_data)

        # go through each bad year for this month
        all_years = np.unique(station.years)
        for year in bad_years:

            # corresponding locations
            ym_locs, = np.where(np.logical_and(station.months == month, station.years == all_years[year]))

            # if pressure or wind speed, need to do some further checking before applying flags
            if obs_var.name in ["station_level_pressure", "sea_level_pressure", "wind_speed"]:

                # pull out the data
                wind_data = station.wind_speed.data[ym_locs]
                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    pressure_data = obs_var.data[ym_locs]
                else:
                    pressure_data = station.sea_level_pressure.data[ym_locs]


                # need sufficient data to work with for storm check to work, else can't tell
                if len(pressure_data.compressed()) < utils.DATA_COUNT_THRESHOLD or \
                        len(wind_data.compressed()) < utils.DATA_COUNT_THRESHOLD:
                    # move on
                    continue

                # find locations of high wind speeds and low pressures, cross match
                high_winds, = np.ma.where((wind_data - wind_average)/wind_spread > STORM_THRESHOLD)
                low_pressures, = np.ma.where((pressure_average - pressure_data)/pressure_spread > STORM_THRESHOLD)

                match = np.in1d(high_winds, low_pressures)

                couldbe_storm = False
                if len(match) > 0:
                    # this could be a storm, either at tropical station (relatively constant pressure)
                    # or out of season in mid-latitudes.
                    couldbe_storm = True

                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    diffs = np.ma.diff(pressure_data)
                elif obs_var.name == "wind_speed":
                    diffs = np.ma.diff(wind_data)

                # count up the largest number of sequential negative and positive differences
                negs, poss = 0, 0
                biggest_neg, biggest_pos = 0, 0

                for diff in diffs:

                    if diff > 0:
                        if negs > biggest_neg: biggest_neg = negs
                        negs = 0
                        poss += 1
                    else:
                        if poss > biggest_pos: biggest_pos = poss
                        poss = 0
                        negs += 1

                if (biggest_neg < 10) and (biggest_pos < 10) and not couldbe_storm:
                    # insufficient to identify as a storm (HadISD values)
                    # leave flags set
                    pass
                else:
                    # could be a storm, so better to leave this month unflagged
                    # zero length array to flag
                    ym_locs = np.ma.array([])


            # copy over the flags, if any
            if len(ym_locs) != 0:
                # and set the flags
                flags[ym_locs] = "V"

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt

            scaled_variances = ((variances - average_variance) / variance_spread)
            bins = utils.create_bins(scaled_variances, 0.25, obs_var.name)
            hist, bin_edges = np.histogram(scaled_variances, bins)

            plt.clf()
            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Months")
            plt.xlabel("Scaled {} Variances".format(obs_var.name.capitalize()))
            plt.title("{} - month {}".format(station.id, month))

            plt.ylim([0.1, max(hist)*2])
            plt.axvline(SPREAD_THRESHOLD, c="r")
            plt.axvline(-SPREAD_THRESHOLD, c="r")

            bad_hist, dummy = np.histogram(scaled_variances[bad_years], bins)
            plt.step(bins[1:], bad_hist, color='r', where="pre")

            plt.show()

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Variance {obs_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")
    if diagnostics:
        print(f"Variance {obs_var.name}")
        print(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    return # variance_check

#************************************************************************
def evc(station: utils.Station, var_list: list, config_dict: dict, full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Excess Variance Check

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        if full:
            find_thresholds(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        variance_check(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)


    return # evc

#************************************************************************
if __name__ == "__main__":

    print("checking excess variance")
#************************************************************************
