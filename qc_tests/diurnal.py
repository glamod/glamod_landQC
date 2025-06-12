"""
Diurnal Cycle Checks
====================

Check whether diurnal cycle is consistent across the record
"""
#************************************************************************
import datetime as dt
import numpy as np
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils

#************************************************************************
# TODO - move these into a config file?
OBS_PER_DAY = 4
DAILY_RANGE = 5

THRESHOLD = 0.33
MISSING = -99

#************************************************************************
def make_sines() -> np.ndarray:
    """
    Return a sine curve spanning y=0 to y=1 . Working in minutes since 00:00

    Returns
    -------
    np.ndarray
        Sine curve for 1440min between 0 & 1
    """

    points = np.arange((24*60), dtype=(np.float64)) / (24.*60.)
    sine_curve = (np.sin(2. * np.pi * points) + 1.)/2.

    # build up an array of 24 sine curves, each one offset by 1hr from previous
    sine = np.copy(sine_curve)
    all_sines = np.zeros((24, sine_curve.shape[0]))
    for h in range(24):
        all_sines[h] = sine
        sine = np.roll(sine, 60)

    return all_sines


#************************************************************************
def quartile_check(minutes: np.ndarray) -> bool:
    """
    Check if >=3 quartiles of the day have data

    :param array minutes: minutes of day where there are data
    :returns: boolean
    """

    quartile_has_data = np.zeros(4)

    quartile_has_data[0] = np.where(np.logical_and(minutes >= 0,
                                                   minutes < 6*60 ))[0].shape[0]
    quartile_has_data[1] = np.where(np.logical_and(minutes >= 6*60,
                                                   minutes < 12*60 ))[0].shape[0]
    quartile_has_data[2] = np.where(np.logical_and(minutes >= 12*60,
                                                   minutes < 18*60 ))[0].shape[0]
    quartile_has_data[3] = np.where(np.logical_and(minutes >= 18*60,
                                                   minutes < 24*60 ))[0].shape[0]

    # binary-ise
    quartile_has_data[quartile_has_data > 0] = 1

    if quartile_has_data.sum() >= 3:
        return True

    else:
        return False # quartile_check


#************************************************************************
def make_scaled_sine(maximum: float,
                     minimum: float) -> np.ndarray:
    """Return sine curve scaled to diurnal range and offset by minmum T

    Parameters
    ----------
    maximum : float
        Diurnal maximum
    minimum : float
        Diurnal minimum

    Returns
    -------
    np.ndarray
        Scaled sine curve for 1440 minutes in a day
    """
    return (make_sines() * (maximum-minimum)) + minimum


#************************************************************************
def find_differences(this_day: np.ndarray,
                     this_day_minutes: np.ndarray) -> np.ndarray:
    """Find differences for each offset hour.

    Parameters
    ----------
    this_day : np.ndarray
        Day of temperature obs
    this_day_minutes : np.ndarray
        Minutes in day with valid obs

    Returns
    -------
    np.ndarray
        Total of differences at each offset
    """
    # a set of sine curves, each one shifted by 1hr from previous
    scaled_sine = make_scaled_sine(np.ma.max(this_day),
                                   np.ma.min(this_day))

    # repeat day 24 times, stacked
    tiled_day = np.tile(this_day, (24, 1))

    # subtract, and then sum squared differences
    differences = np.ma.sum((
        tiled_day - scaled_sine[:, this_day_minutes]
        )**2, axis=1)

    return differences


#************************************************************************
def find_uncertainty(differences: np.ndarray, best_fit: int) -> int:
    """Find the uncertainty around the best fit diurnal

    Parameters
    ----------
    differences : np.ndarray
        Array of the differences of theoretical sines to data,
        one for each hr shift of 24
    best_fit : int
        Location of minimum difference

    Returns
    -------
    int
        Size of uncertainty in best fit as determined by
        threshold [0.33] of total range of differences
    """

    # now to get uncertainties on this best fit shift
    critical_value = np.ma.min(differences) +\
        ((np.ma.max(differences) - np.ma.min(differences))*THRESHOLD)

    # roll, so best fit is in the middle
    differences = np.roll(differences, (11 - best_fit))

    """
     |                           uncertainty
     |                            |------|
     |             ----
 D   |   *  *  *    |                                  *  *  *
 i   |            * | *                          *  *
 f   |              |    *                    *
 f   |        0.66  |       *              *
 s   |--------------------------------------------------------
     |        0.33  |          *        *
     |              |
     |             ----           *  *  <---- Min Diff
     |_______________________________________________________
                              HOURS
    """


    # find where below critical, add 1 to max offset
    locs, = np.where(differences < critical_value)
    uncertainty = 1+ np.max([11-locs[0],locs[-1]-11])

    return uncertainty

#************************************************************************
def find_fit_and_uncertainty(this_day: np.ndarray,
                             this_day_minutes: np.ndarray) -> tuple[int, int]:
    """
    Find the best fit of a theoretical sine curve to the data.
    Shift theoretical sine curve by hours and getting smallest offset
    Unlikely to be able to use information on a smaller scale than hours
    even though the data is per minutes.

    :param array this_day: data for day that have obs
    :param array this_day_minutes: minutes for day which have obs [index 0-1439]

    :returns: best_fit, uncertainty
    """

    differences = find_differences(this_day, this_day_minutes)

    # location of the best fit
    best_fit = np.argmin(differences)

    uncertainty = find_uncertainty(differences, best_fit)

    return best_fit, uncertainty

#************************************************************************
def get_daily_offset(station: utils.Station, locs: np.ndarray,
                     obs_var: utils.Meteorological_Variable) -> tuple[int, int]:
    """
    Extract data for a single 24h period, and pass to offset finder

    :param Station station: station object for the station
    :param array locs: locations corresponding for selected day
    :param MetVar obs_var: Meteorological variable object

    :returns: best_fit, uncertainty - ints
    """

    best_fit, uncertainty = MISSING, MISSING

    # identify each day
    this_day = obs_var.data[locs]

    # further restrictions (range>=5K, at least in 3 of 4 quarters of the day etc)
    if len(this_day.compressed()) > OBS_PER_DAY:
        if np.ma.max(this_day) - np.ma.min(this_day) > DAILY_RANGE:

            these_times = station.times[locs] - station.times[locs].iloc[0]
            this_day_mins = (these_times.to_numpy()/np.timedelta64(1, "m")).astype(int)
            print(this_day_mins)
            if quartile_check(this_day_mins):
                best_fit, uncertainty = find_fit_and_uncertainty(this_day,
                                                                 this_day_mins)

    return best_fit, uncertainty # get_daily_offset

#************************************************************************
def prepare_data(station: utils.Station,
                 obs_var: utils.Meteorological_Variable) -> tuple[np.ndarray, np.ndarray]:
    """
    For each 24h period, find diurnal cycle offset and uncertainty

    :param Station station: station object for the station
    :param MetVar obs_var: Meteorological variable object

    :returns: best_fit, uncertainty - arrays
    """

    ndays = dt.date(station.times.iloc[-1].year + 1, 1, 1) - dt.date(station.times.iloc[0].year, 1, 1)
    best_fit_diurnal = np.ones(ndays.days + 1).astype(int)
    best_fit_uncertainty = np.zeros(ndays.days + 1).astype(int)
    d = 0
    for year in np.unique(station.years):
        for month in np.unique(station.months):
            for day in np.unique(station.days):
                try:
                    # if Datetime doesn't throw an error, then valid date
                    _ = dt.datetime(year, month, day)
                except ValueError:
                    # not a valid day (e.g. Leap years, short months etc)
                    continue

                locs, = np.where(np.logical_and.reduce((station.years == year, station.months == month, station.days == day)))

                if len(locs) > OBS_PER_DAY:
                    # at least have the option of enough data
                    best_fit_diurnal[d], best_fit_uncertainty[d] = get_daily_offset(station, locs, obs_var)
                else:
                    best_fit_diurnal[d], best_fit_uncertainty[d] = MISSING, MISSING
                # and move on to the next day
                d += 1

    return best_fit_diurnal, best_fit_uncertainty # prepare_data

#************************************************************************
def find_offset(obs_var: utils.Meteorological_Variable, station: utils.Station, config_dict: dict, plots: bool = False, diagnostics: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the best offset for a sine curve to represent the cycle

    :param MetVar obs_var: Meteorological Variable object
    :param Station station: Station Object for the station
    :param str config_dict: dictionary containing configuration info
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

        if len(locs) >= utils.DATA_COUNT_THRESHOLD:
            best_fits[h] = np.median(best_fit_diurnal[locs])

    # now go through each of the 6hrs of uncertainty and see if the range
    # of the best fit +/- uncertainty overlap across them.
    # if they do, it's a well defined cycle, if not, then there's a problem

    '''Build up range of cycles incl, uncertainty to find where best of best located'''

    hours = np.arange(24)
    hour_matches = np.zeros(24)
    diurnal_peak = MISSING
    number_estimates = 0
    for h in range(6):
        if best_fits[h] != MISSING:
            '''Store lowest uncertainty best fit as first guess'''
            if diurnal_peak == MISSING:
                diurnal_peak = best_fits[h]
                hours = np.roll(hours, 11-int(diurnal_peak))
                hour_matches[11-(h+1):11+(h+2)] = 1
                number_estimates += 1

            # get spread of uncertainty, and +1 to this range
            centre, = np.where(hours == best_fits[h])

            if (centre[0] - (h + 1)) >= 0:
                if (centre[0] + h + 1) <= 23:
                    hour_matches[centre[0] - (h + 1) : centre[0] + (h + 2)] += 1
                else:
                    hour_matches[centre[0] - (h + 1) : ] += 1 # back part
                    hour_matches[ : centre[0] + (h + 2) - 24] += 1 # front part
            else:
                hour_matches[: centre[0] + h + 2] += 1 # front part
                hour_matches[centre[0] - (h + 1) :] += 1 # back part

            number_estimates += 1

    '''If value at lowest uncertainty not found in all others, then see what value is found by all others '''
    if hour_matches[11] != number_estimates:  # central estimate at 12 o'clock
        all_match, = np.where(hour_matches == number_estimates)

        # if one is, then use it
        if len(all_match) > 0:
            diurnal_peak = all_match[0]
        else:
            diurnal_peak = MISSING
            logger.warning("Good fit to diurnal cycle not found")

    '''Now have value for best fit diurnal offset'''
    CD_peak = {"peak" : int(diurnal_peak)}
    config_dict[f"DIURNAL-{obs_var.name}"] = CD_peak

    return best_fit_diurnal, best_fit_uncertainty # find_offset

#************************************************************************
def diurnal_cycle_check(obs_var: utils.Meteorological_Variable, station: utils.Station, config_dict: dict,
                        plots: bool = False, diagnostics: bool = False, best_fit_diurnal: np.ndarray = None,
                        best_fit_uncertainty: np.ndarray = None) -> None:
    """
    Use offset to find days where cycle doesn't match

    :param MetVar obs_var: Meteorological Variable object
    :param Station station: Station Object for the station
    :param str config_dict: dictionary to store configuration info
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """
    flags = np.array(["" for i in range(obs_var.data.shape[0])])

    try:
        diurnal_offset = int(config_dict[f"DIURNAL-{obs_var.name}"]["peak"])
    except KeyError:
        print("Information missing in config dictionary")
        best_fit_diurnal, best_fit_uncertainty = find_offset(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        diurnal_offset = int(config_dict[f"DIURNAL-{obs_var.name}"]["peak"])


    hours = np.arange(24)
    hours = np.roll(hours, 11-int(diurnal_offset))


    if diurnal_offset != MISSING:

        if (best_fit_diurnal is None) and (best_fit_uncertainty is None):
            best_fit_diurnal, best_fit_uncertainty = prepare_data(station, obs_var)

        # find locations where the overall best fit does not match the daily fit
        potentially_spurious = np.ones(best_fit_diurnal.shape[0])*MISSING

        for d, (fit, uncertainty) in enumerate(zip(best_fit_diurnal, best_fit_uncertainty)):
            if fit != MISSING:
                min_range = 11 - uncertainty
                max_range = 11 + uncertainty

                offset_loc, = np.where(hours == fit)

                # find where the best fit falls outside the range for this particular day
                if offset_loc < min_range or offset_loc > max_range:
                    potentially_spurious[d] = 1
                else:
                    potentially_spurious[d] = 0

        # now check there are sufficient issues in running 30 day periods
        """Any periods>30 days where the diurnal cycle deviates from the expected
        phase by more than this uncertainty, without three consecutive good or missing days
        or six consecutive days consisting of a mix of only good or missing values, a
        re deemed dubious and the entire period of data (including all non-temperature elements) is flagged"""
        # here only temperature elements are flagged

        n_good = 0
        n_miss = 0
        n_not_bad = 0
        total_points = 0
        total_not_miss = 0
        bad_locs = np.zeros(best_fit_diurnal.shape[0])

        for d in range(best_fit_diurnal.shape[0]):

            if potentially_spurious[d] == 1:
                # if bad, just add one
                n_good = 0
                n_miss = 0
                n_not_bad = 0
                total_points += 1
                total_not_miss += 1

            else:
                # find a non-bad value - so check previous run
                #  if have reached limits on good/missing
                if (n_good == 3) or (n_miss == 3) or (n_not_bad >= 6):
                    # sufficient good missing or not bad data
                    if total_points >= 30:
                        # if have collected enough others, then set flag
                        if float(total_not_miss)/total_points >= 0.5:
                            bad_locs[d - total_points : d] = 1
                    # reset counters
                    n_good = 0
                    n_miss = 0
                    n_not_bad = 0
                    total_points = 0
                    total_not_miss = 0

                # and deal with this point
                total_points += 1
                if potentially_spurious[d] == 0:
                    # if good
                    n_good += 1
                    n_not_bad += 1
                    if n_miss != 0:
                        n_miss = 0
                    total_not_miss += 1

                elif potentially_spurious[d] == -999:
                    # if missing data
                    n_miss += 1
                    n_not_bad += 1
                    if n_good != 0:
                        n_good = 0

        # run through all days
        # find zero point of day counter in data preparation part
        day_counter_start = dt.datetime(np.unique(station.years)[0],
                                        np.unique(station.months)[0],
                                        np.unique(station.days)[0])

        # find the bad days in the times array
        for day, bad in enumerate(bad_locs):
            if bad == 0:
                # good days don't need processing
                continue
            this_day = day_counter_start + dt.timedelta(days=int(day))

            locs, = np.where(np.logical_and.reduce((station.years == this_day.year,
                                                    station.months == this_day.month,
                                                    station.days == this_day.day)))

            # only set flag on where there's data
            data_locs, = np.where(obs_var.data[locs].mask == False)

            flags[locs[data_locs]] = "U"

        # append flags to object
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        logger.info(f"Diurnal Check {obs_var.name}")
        logger.info(f"   Cumulative number of flags set: {len(np.where(flags != '')[0])}")

    else:
        logger.info("Diurnal fit not found")

    return # diurnal_cycle_check


#************************************************************************
def dcc(station: utils.Station, config_dict: dict, full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Pass on to the Diurnal Cycle Check

    :param Station station: Station Object for the station
    :param str config_dict: dictionary contining configuration info
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    obs_var = getattr(station, "temperature")

    if full:
        best_fit_diurnal, best_fit_uncertainty = find_offset(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)
        diurnal_cycle_check(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics, \
                                best_fit_diurnal=best_fit_diurnal, best_fit_uncertainty=best_fit_uncertainty)

    else:
        diurnal_cycle_check(obs_var, station, config_dict, plots=plots, diagnostics=diagnostics)

    return # dgc

