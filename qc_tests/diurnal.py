"""
Diurnal Cycle Checks
^^^^^^^^^^^^^^^^^^^^^^^^^

Check whether diurnal cycle is consistent across the record
"""
#************************************************************************
import sys
import numpy as np
import scipy as sp
import datetime as dt
import calendar

import qc_utils as utils

#************************************************************************
# TODO - move these into a config file?
OBS_PER_DAY = 4
DAILY_RANGE = 5

THRESHOLD = 0.33
MISSING = -99

# sine curve spanning y=0 to y=1 . Working in minutes since 00:00
SINE_CURVE = (np.sin(2.*np.pi* np.arange((24*60),dtype=(np.float64)) / (24.*60.)) + 1.)/2.


#************************************************************************
def quartile_check(minutes):
    """
    Check if >=3 quartiles of the day have data
    
    :param array minutes: minutes of day where there are data
    :returns: boolean
    """
    
    quartile_has_data = np.zeros(4)

    for minute in minutes:

        if minute > 0 and minute < 6*60:
            quartile_has_data[0] = 1
        elif minute > 6*60 and minute < 12*60:
            quartile_has_data[1] = 1
        elif minute > 12*60 and minute < 18*60:
            quartile_has_data[2] = 1
        elif minute > 18*60 and minute < 24*60:
            quartile_has_data[3] = 1

    if quartile_has_data.sum() >= 3:
        return True

    else:
        return False # quartile_check

#************************************************************************
def find_fit(this_day, this_day_mins):
    """
    Find the best fit of a theoretical sine curve to the data

    :param array this_day: data for day that have obs
    :param array this_day_mins: minutes for day which have obs [index 0-1439]

    :returns: best_fit, uncertainty
    """

    diurnal_range = np.ma.max(this_day) - np.ma.min(this_day)

    scaled_sine = (SINE_CURVE * diurnal_range) + np.ma.min(this_day)

    # find difference for each successive hour shift
    #    data is in minutes, but only shifting hours as unlikely to resolve
    #    the cycle any more finely than this.....???
    differences = np.zeros(24)
    for h in range(24):
        differences[h] = np.ma.sum(np.abs(this_day - scaled_sine[this_day_mins]))
        scaled_sine = np.roll(scaled_sine, 60)

    best_fit = np.argmin(differences)

    # now to get uncertainties on this best fit shift
    critical_value = np.ma.min(differences) + (np.ma.max(differences) - np.ma.min(differences))*THRESHOLD
    # roll, so best fit is in the middle
    differences = np.roll(differences, (11 - best_fit))

    # head up the sides (out of the well of the minimum) 
    #     and get the spread once greater than critical
    """
     |                                                         
     |                             uncertainty                
     |                            |-----------|                
     |                                                         
 D   |   *  *  *                                      *  *  *  
 i   |            *  *                          *  *           
 f   |                  *                    *                 
 f   |                     *              *                    
 s   |                                                         
     |                        *        *                       
     |                                                         
     |                                                         
     |                           *  *  <---- Min Diff                
     |                                                         
     |                                                         
     |                                                         
     |                                                         
     |_________________________________________________________________
                              HOURS
    """             
    uncertainty = 1 # hours
    while uncertainty < 11: # i.e. undefined
        if differences[11-uncertainty] > critical_value and\
           differences[11+uncertainty] > critical_value:
            break
        uncertainty += 1

    return best_fit, uncertainty

#************************************************************************
def get_daily_offset(station, locs, obs_var):
    """
    Extract data for a single 24h period, and pass to offset finder

    :param Station station: station object for the station
    :param array locs: locations corresponding for selected day
    :param MetVar obs_var: Meteorological variable object

    :returns: best_fit, uncertainty - ints
    """

    # identify each day
    this_day = obs_var.data[locs]
    these_times = station.times[locs] - station.times[locs].iloc[0]
    this_day_mins = (these_times.to_numpy()/np.timedelta64(1, "m")).astype(int)
    
    # TODO - further restrictions (range>=5K, at least in 3 of 4 quarters of the day etc)
    best_fit, uncertainty = MISSING, MISSING
    if len(this_day.compressed()) > OBS_PER_DAY:
        if np.ma.max(this_day) - np.ma.min(this_day) > DAILY_RANGE:
            if quartile_check(this_day_mins):            
                best_fit, uncertainty = find_fit(this_day, this_day_mins)
    
    return best_fit, uncertainty # get_daily_offset

#************************************************************************
def prepare_data(station, obs_var):
    """
    For each 24h period, find diurnal cycle offset and uncertainty

    :param Station station: station object for the station
    :param MetVar obs_var: Meteorological variable object

    :returns: best_fit, uncertainty - arrays
    """

    ndays = station.times.iloc[-1] - station.times.iloc[0]
    best_fit_diurnal = np.zeros(ndays.days + 1).astype(int)
    best_fit_uncertainty = np.zeros(ndays.days + 1).astype(int)
    d = 0
    for year in np.unique(station.years):
        for month in np.unique(station.months):
            for day in np.unique(station.days):

                try:
                    dummy = dt.datetime(year, month, day)
                except ValueError:
                    # not a valid day (e.g. Leap years, short months etc)
                    continue

                locs, = np.where(np.logical_and.reduce((station.years == year, station.months == month, station.days == day)))

                if len(locs) != 0:
                    best_fit_diurnal[d], best_fit_uncertainty[d] = get_daily_offset(station, locs, obs_var)

                # and move on to the next day     
                d += 1

    return best_fit_diurnal, best_fit_uncertainty # prepare_data 

#************************************************************************
def find_offset(obs_var, station, config_file, plots=False, diagnostics=False):
    """
    Find the best offset for a sine curve to represent the cycle

    :param MetVar obs_var: Meteorological Variable object
    :param Station station: Station Object for the station
    :param str configfile: string for configuration file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    best_fit_diurnal, best_fit_uncertainty = prepare_data(station, obs_var)
    
    # done complete record, have best fit for each day
    # now to find best overall fit.
    #    find median offset for each uncertainty range from 1 to 6 hours
    best_fits = MISSING*np.ones(6).astype(int)
    for h in range(6):
        locs, = np.where(best_fit_uncertainty == h+1)
        # TODO - move 300 to config or header
        if len(locs) >= 300:
            best_fits[h] = np.median(best_fit_diurnal[locs])

    # now go through each of the 6hrs of uncertainty and see if the range
    # of the best fit +/- uncertainty overlap across them.
    # if they do, it's a well defined cycle, if not, then there's a problem

    '''Build up range of cycles incl, uncertainty to find where best of best located'''

    hours = np.arange(24)
    hour_matches=np.zeros(24)
    diurnal_peak = MISSING
    number_estimates = 0
    for h in range(6):
        if best_fits[h] != MISSING:
            '''Store lowest uncertainty best fit as first guess'''
            if diurnal_peak == MISSING: 
                diurnal_peak = best_fits[h]
                hours = np.roll(hours,11-int(diurnal_peak))
                hour_matches[11-(h+1):11+(h+2)] = 1
                number_estimates += 1

            # get spread of uncertainty, and +1 to this range 
            centre, = np.where(hours == best_fits[h])

            if (centre[0] - h + 1) >= 0:
                if (centre[0] + h + 1 ) <=23:
                    hour_matches[centre[0] - (h + 1) : centre[0] + h + 2] += 1
                else:
                    hour_matches[centre[0] - (h + 1) : ] += 1
                    hour_matches[ : centre[0] + h + 2- 24] += 1                                        
            else:
                hour_matches[: centre[0] + h + 2] += 1
                hour_matches[centre[0] - (h + 1) :] += 1

            number_estimates += 1

    '''If value at lowest uncertainty not found in all others, then see what value is found by all others '''
    if hour_matches[11] != number_estimates:  # central estimate at 12 o'clock
        all_match, = np.where(hour_matches == number_estimates)

        # if one is, then use it
        if len(all_match[0]) > 0:
            diurnal_peak = all_match[0]
        else:
            diurnal_peak = MISSING

    '''Now have value for best fit diurnal offset'''
    utils.write_qc_config(config_file, "DIURNAL-{}".format(obs_var.name), "peak", "{}".format(diurnal_peak), diagnostics=diagnostics)

    return # find_offset

#************************************************************************
def diurnal_cycle_check(obs_var, station, config_file, plots=False, diagnostics=False):
    """
    Use offset to find days where cycle doesn't match

    :param MetVar obs_var: Meteorological Variable object
    :param Station station: Station Object for the station
    :param str configfile: string for configuration file
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    diurnal_offset = int(utils.read_qc_config(config_file, "DIURNAL-{}".format(obs_var.name), "peak"))

    if diurnal_offset != MISSING:

        best_fit_diurnal, best_fit_uncertainty = prepare_data(station, obs_var)

        # get the range of possible values allowed for each day
        min_range = best_fit_diurnal - best_fit_uncertainty
        min_range[min_range < 0] = min_range[min_range < 0] + 24

        max_range = best_fit_diurnal + best_fit_uncertainty
        max_range[max_range >= 24] = max_range[max_range >= 24] - 24
                
        # check if global best fit falls inside daily fit +/- uncertainty
        bad_locs, = np.where(np.logical_and(min_range >= diurnal_offset, max_range <= diurnal_offset))

        # run through all days
        # find zero point of day counter in data preparation part
        day_counter_start = dt.datetime(np.unique(station.years)[0], np.unique(station.months)[0], np.unique(station.days)[0])

        # find the bad days in the times array
        for day in bad_locs:

            this_day = day_counter_start + dt.timedelta(days = int(day))

            locs, = np.where(np.logical_and.reduce((station.years == this_day.year, station.months == this_day.month, station.days == this_day.day)))

            flags[locs] = "U"

        # append flags to object
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        if diagnostics:

            print("Diurnal Check {}".format(obs_var.name))
            print("   Cumulative number of flags set: {}".format(len(np.where(flags != "")[0])))
            
    return # diurnal_cycle_check


#************************************************************************
def dcc(station, config_file, full=False, plots=False, diagnostics=False):
    """
    Pass on to the Diurnal Cycle Check

    :param Station station: Station Object for the station
    :param str configfile: string for configuration file
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    obs_var = getattr(station, "temperature")

    if full:
        find_offset(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)

    diurnal_cycle_check(obs_var, station, config_file, plots=plots, diagnostics=diagnostics)


    return # dgc

#************************************************************************
if __name__ == "__main__":
    
    print("checking diurnal cycle")
#************************************************************************
