"""
Distributional Gap Checks
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Check distribution of monthly values and look for assymmetry
2. Check distribution of all observations and look for distinct populations
"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
from scipy.stats import skew
import datetime as dt
import copy

import qc_utils as utils
#************************************************************************
STORM_THRESHOLD = 5

OBS_LIMIT = 50 
VALID_MONTHS = 5
MIN_OBS = 28*2 # two obs per day for the month (minimum subdaily)
SPREAD_LIMIT = 2 # two IQR/MAD/STD
BIN_WIDTH = 0.25
LARGE_LIMIT = 5
GAP_SIZE = 2
FREQUENCY_THRESHOLD = 0.1
#************************************************************************
def prepare_monthly_data(obs_var, station, month, diagnostics=False):
    """
    Extract monthly data and make average.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param bool diagnostics: turn on diagnostic output
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
def find_monthly_scaling(obs_var, station, config_file, diagnostics=False):
    """
    Find scaling parameters for monthly values and store in config file

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_file: configuration file to store critical values
    :param bool diagnostics: turn on diagnostic output
    """

    all_years = np.unique(station.years)
    
    for month in range(1, 13):

        month_averages = prepare_monthly_data(obs_var, station, month, diagnostics = diagnostics)

        if len(month_averages.compressed()) >= VALID_MONTHS:

            # have months, now to standardise
            climatology = utils.average(month_averages) # mean
            spread = utils.spread(month_averages) # IQR currently
            if spread < SPREAD_LIMIT:
                spread = SPREAD_LIMIT

            # write out the scaling...
            utils.write_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-clim".format(month), "{}".format(climatology), diagnostics=diagnostics)
            utils.write_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-spread".format(month), "{}".format(spread), diagnostics=diagnostics)

        else:
            utils.write_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-clim".format(month), "{}".format(utils.MDI), diagnostics=diagnostics)
            utils.write_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-spread".format(month), "{}".format(utils.MDI), diagnostics=diagnostics)


    return # find_monthly_scaling

#************************************************************************
def monthly_gap(obs_var, station, config_file, plots=False, diagnostics=False):
    """
    Use distribution to identify assymetries.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])
    all_years = np.unique(station.years)
    
    for month in range(1, 13):

        month_averages = prepare_monthly_data(obs_var, station, month, diagnostics = diagnostics)

        # read in the scaling
        climatology = float(utils.read_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-clim".format(month)))
        spread = float(utils.read_qc_config(config_file, "MDISTRIBUTION-{}".format(obs_var.name), "{}-spread".format(month)))

        if climatology == utils.MDI and spread == utils.MDI:
            # these weren't calculable, move on
            continue

        standardised_months = (month_averages - climatology) / spread

        bins = utils.create_bins(standardised_months, BIN_WIDTH)
        hist, bin_edges = np.histogram(standardised_months, bins)

        if plots:
            import matplotlib.pyplot as plt
            
            plt.step(bins[1:], hist, color='k', where="pre")

            plt.ylabel("Number of Months")
            plt.xlabel(obs_var.name.capitalize())
            plt.title("{} - month {}".format(station.id, month))

            plt.show()       

        # flag months with very large offsets
        bad, = np.where(np.abs(standardised_months) >= LARGE_LIMIT)

        # walk distribution from centre to find assymetry
        sort_order = standardised_months.argsort()
        mid_point = len(standardised_months) // 2
        good = True
        step = 1
        while good:

            if standardised_months[sort_order][mid_point - step] != standardised_months[sort_order][mid_point + step]:
                
                suspect_months = [standardised_months[sort_order][mid_point - step], standardised_months[sort_order][mid_point + step]]
                
                if min(suspect_months) != 0:
                    # not all clustered at origin
                    
                    if max(suspect_months)/min(suspect_months) >= 2. and min(suspect_months) >= 1.5:
                        # at least 1.5x spread from centre and difference of two in location (longer tail)
                        # flag everything further from this bin for that tail
                        if suspect_months[0] == max(suspect_months):
                            # LHS has issue
                            bad = sort_order[:mid_point - iter]
                        elif suspect_months[1] == max(suspect_months):
                            # RHS has issue
                            bad = sort_order[mid_point + iter:]

            step += 1
            if step == mid_point:
                # reached end
                break
    
        # now follow flag locations back up through the process
        for bad_month_id in bad:
             # year ID for this set of calendar months
             for year in all_years:
                 if year in bad_month_id:
                     locs, = np.where(np.logical_and(station.months == month, station.years == year))
                     flags[locs] = "D"

    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Distribution (monthly) {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # monthly_gap

#************************************************************************
def prepare_all_data(obs_var, station, month, config_file, full=False, diagnostics=False):
    """
    Extract data for the month, make & store or read average and spread.
    Use to calculate normalised anomalies.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param str config_file: configuration file to store critical values
    :param bool diagnostics: turn on diagnostic output
    """

    month_locs, = np.where(station.months == month)

    all_month_data = obs_var.data[month_locs]

    if full:

        if len(all_month_data.compressed()) > OBS_LIMIT:
            # have data, now to standardise
            climatology = utils.average(all_month_data) # mean
            spread = utils.spread(all_month_data) # IQR currently
        else:
            climatology = utils.MDI
            spread = utils.MDI

        # write out the scaling...
        utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-clim".format(month), "{}".format(climatology), diagnostics=diagnostics)
        utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-spread".format(month), "{}".format(spread), diagnostics=diagnostics)
        
    else:
        climatology = float(utils.read_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-clim".format(month)))
        spread = float(utils.read_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-spread".format(month)))        

    if climatology == utils.MDI and spread == utils.MDI:
        # these weren't calculable, move on
        return np.ma.array([utils.MDI])
    else:
        return (all_month_data - climatology)/spread  # prepare_all_data

#************************************************************************
def find_thresholds(obs_var, station, config_file, plots=False, diagnostics=False):
    """
    Extract data for month and find thresholds in distribution and store.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param int month: month to process
    :param str config_file: configuration file to store critical values
    :param bool diagnostics: turn on diagnostic output
    """

  
    for month in range(1, 13):

        normalised_anomalies = prepare_all_data(obs_var, station, month, config_file, full=True, diagnostics=diagnostics)

        if len(normalised_anomalies.compressed()) == 1 and normalised_anomalies[0] == utils.MDI:
            # scaling not possible for this month
            utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-uthresh".format(month), "{}".format(utils.MDI), diagnostics=diagnostics)
            utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-lthresh".format(month), "{}".format(utils.MDI), diagnostics=diagnostics)
            continue

        bins = utils.create_bins(normalised_anomalies, BIN_WIDTH)
        hist, bin_edges = np.histogram(normalised_anomalies, bins)

        gaussian_fit = utils.fit_gaussian(bins[1:], hist, max(hist), mu = bins[np.argmax(hist)], sig = utils.spread(normalised_anomalies), skew = skew(normalised_anomalies.compressed()))

        fitted_curve = utils.skew_gaussian(bins[1:], gaussian_fit)

        # diagnostic plots
        if plots:
            import matplotlib.pyplot as plt
            plt.step(bins[1:], hist, color='k', where="pre")
            plt.yscale("log")

            plt.ylabel("Number of Observations")
            plt.xlabel(obs_var.name.capitalize())
            plt.title("{} - month {}".format(station.id, month))

            plt.plot(bins[1:], fitted_curve)
            plt.ylim([0.1, max(hist)*2])

        # use bins and curve to find points where curve is < FREQUENCY_THRESHOLD
        try:
            lower_threshold = bins[1:][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bins[1:] < 0))[0]][-1]
        except:
            lower_threshold = bins[1]
        try:
            upper_threshold = bins[1:][np.where(np.logical_and(fitted_curve < FREQUENCY_THRESHOLD, bins[1:] > 0))[0]][0]
        except:
            upper_threshold = bins[-1]

        if plots:
            plt.axvline(upper_threshold, c="r")
            plt.axvline(lower_threshold, c="r")
            plt.show()
            
        utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-uthresh".format(month), "{}".format(upper_threshold), diagnostics=diagnostics)
        utils.write_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-lthresh".format(month), "{}".format(lower_threshold), diagnostics=diagnostics)

    return # find_thresholds

#************************************************************************
def find_gap(hist, bins, threshold, upwards=True, gap_size=GAP_SIZE):
    '''
    Walk the bins of the distribution to find a gap and return where it starts
   
    :param array hist: histogram values
    :param array bins: bin values
    :param flt threshold: limiting value
    :param int gap_size: gap size to record
    :returns:
        flt: gap_start
    '''

    # start in the centre
    start = np.argmax(hist)
       
    n = 0
    gap_length = 0
    gap_start = 0
    while True:
        # if bin is zero - could be a gap
        if hist[start + n] == 0:
            gap_length += 1
            if gap_start == 0:
                # plus 1 to get upper bin boundary
                if (upwards and bins[start + n + 1] >= threshold):
                    gap_start = bins[start + n + 1]
                elif (not upwards and bins[start + n] <= threshold):
                    gap_start = bins[start + n]
                
        # bin has value
        else:
            # gap too short
            if gap_length < gap_size:
                gap_length = 0
                
            # found a gap
            elif gap_length >= gap_size and gap_start != 0:
                break
        # escape if gone off the end of the distribution
        if (start + n == len(hist) - 1) or (start + n == 0):
            break
        
        # increment counters
        if upwards:
            n += 1
        else:
            n -= 1

    return gap_start # find_gap

#************************************************************************
def expand_around_storms(storms, maximum, pad=6):
    """
    Pad storm signal by N=6 hours

    """
    for i in range(pad):

        if storms[0]-1 >=0:
            storms = np.insert(storms, 0, storms[0]-1)
        if storms[-1]+1 < maximum:
            storms = np.append(storms, storms[-1]+1)

    return np.unique(storms) # expand_around_storms

#************************************************************************
def all_obs_gap(obs_var, station, config_file, plots=False, diagnostics=False):
    """
    Extract data for month and find secondary populations in distribution.

    :param MetVar obs_var: meteorological variable object
    :param Station station: station object
    :param str config_file: configuration file to store critical values
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    flags = np.array(["" for i in range(obs_var.data.shape[0])])
    
    for month in range(1, 13):

        normalised_anomalies = prepare_all_data(obs_var, station, month, config_file, full=False, diagnostics=diagnostics)

        bins = utils.create_bins(normalised_anomalies, BIN_WIDTH)
        hist, bin_edges = np.histogram(normalised_anomalies, bins)

        upper_threshold = float(utils.read_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-uthresh".format(month)))
        lower_threshold = float(utils.read_qc_config(config_file, "ADISTRIBUTION-{}".format(obs_var.name), "{}-lthresh".format(month)))

        if upper_threshold == utils.MDI and lower_threshold == utils.MDI:
            # these weren't able to be calculated, move on
            continue

        # now to find the gaps
        uppercount = len(np.where(normalised_anomalies > upper_threshold)[0])
        lowercount = len(np.where(normalised_anomalies < lower_threshold)[0])
        
        month_locs, = np.where(station.months == month) # append should keep year order
        if uppercount > 0:
            gap_start = find_gap(hist, bins, upper_threshold)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies > gap_start) # all years for one month

                month_flags = np.array(["" for i in range(month_locs.shape[0])])
                month_flags[bad_locs] = "d"
                flags[month_locs] = month_flags
                                       
        if lowercount > 0:
            gap_start = find_gap(hist, bins, lower_threshold, upwards=False)

            if gap_start != 0:
                bad_locs, = np.ma.where(normalised_anomalies < gap_start) # all years for one month

                month_flags = np.array(["" for i in range(month_locs.shape[0])])
                month_flags[bad_locs] = "d"

                # TODO - can this bit be refactored
                # for pressure data, see if the flagged obs correspond with high winds
                # could be a storm signal
                if obs_var.name in ["station_level_pressure", "sea_level_pressure"]:
                    wind_monthly_data = prepare_monthly_data(station.wind_speed, station, month)
                    wind_monthly_average = utils.average(wind_monthly_data)
                    wind_monthly_spread = utils.spread(wind_monthly_data)

                    pressure_monthly_data = prepare_monthly_data(obs_var, station, month)
                    pressure_monthly_average = utils.average(pressure_monthly_data)
                    pressure_monthly_spread = utils.spread(pressure_monthly_data)

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
                            separations, = np.where(storm_1diffs > np.median(np.diff(wind_data)))

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
                        
    # append flags to object
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    if diagnostics:
        
        print("Distribution (all) {}".format(obs_var.name))
        print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))

    return # all_obs_gap

#************************************************************************
def dgc(station, var_list, config_file, full=False, plots=False, diagnostics=False):
    """
    Run through the variables and pass to the Distributional Gap Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)

        # monthly gap
        if full:
            find_monthly_scaling(obs_var, station, config_file, diagnostics=diagnostics)
        monthly_gap(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)
        
        # all observations gap
        if full:
            find_thresholds(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)
        all_obs_gap(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)


    return # dgc

#************************************************************************
if __name__ == "__main__":
    
    print("checking gaps in distributions")
#************************************************************************
