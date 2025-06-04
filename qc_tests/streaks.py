"""
Repeated Streaks Check
======================

   Checks for replication of:

     1. checks for consecutive repeating values
              [get_repeating_streak_threshold(), repeating_value()]
     2. checks if one year has more repeating streaks than expected
              [get_excess_streak_threshold(), excess_repeating_value()]
     3. checks for repeats at a given hour across a number of days
              [Not yet implemented]
     4. checks for repeats for whole days - sequences of days where all values repeat
              [repeating_day() - used to set thresholds and apply flagging]

   Thresholds now determined dynamically.
"""
#************************************************************************
import copy
import numpy as np
import datetime as dt
import logging
logger = logging.getLogger(__name__)

import qc_utils as utils

MIN_STREAK_LENGTH_FOR_EXCESS_FREQUENCY = 10

#*********************************************
def plot_streak(times: np.ndarray, data: np.ndarray, units: str,
                streak_start: int, streak_end: int) -> None:
    '''
    Plot each streak against surrounding data

    :param array times: datetime array
    :param array data: values array
    :param str units: units for plotting
    :param int streak_start: the location of the streak
    :param int streak_end: the end of the streak

    :returns:
    '''
    import matplotlib.pyplot as plt

    pad_start = streak_start - 48
    if pad_start < 0:
        pad_start = 0
    pad_end = streak_end + 48
    if pad_end > len(data.compressed()):
        pad_end = len(data.compressed())

    # simple plot
    plt.clf()
    plt.plot(times.compressed()[pad_start: pad_end],
             data.compressed()[pad_start: pad_end], 'ko', )
    plt.plot(times.compressed()[streak_start: streak_end],
             data.compressed()[streak_start: streak_end], 'ro')

    plt.ylabel(units)
    plt.show()

    # plot_streak


def mask_calms(this_var: utils.Meteorological_Variable) -> None:
    """
    Mask calm periods (wind speed == 0) as these can be a legitimate streak
    of repeating values.

    :param MetVar this_var: variable to process (wind speeds)
    """
    reporting_resolution = utils.reporting_accuracy(this_var.data)

    # Don't take an identically equal to 0m/s given reporting
    #   resolution (even if good) may get the odd 0.5m/s as part of a long
    #   calm spell
    if reporting_resolution > 0.5: # nearest 1/2 or whole value
        calms, = np.ma.where(this_var.data <= 1.0)
    else: # nearest 1/10
        calms, = np.ma.where(this_var.data <= 0.5)

    this_var.data[calms] = utils.MDI
    this_var.data.mask[calms] = True

    # mask_calms


def prepare_obs_var(obs_var: utils.Meteorological_Variable,
                    wind_speed: utils.Meteorological_Variable | None = None)\
                    -> utils.Meteorological_Variable:
    """
    For all these checks make a copy of the observational variable so
    masks can be applied without impacting other tests

    And optionally for wind speed and direction, mask out calm periods
    as these could be a true streak

    :param MetVar obs_var: meteorological variable object
    :param MetVar wind_speeds: meteorological variable object

    :returns: MetVar
    """
    this_var = copy.deepcopy(obs_var)

    if obs_var.name == "wind_speed":
        mask_calms(this_var)
    elif obs_var.name == "wind_direction":
        mask_calms(wind_speed)
        # combine the masks
        this_var.data.mask = np.ma.mask_or(wind_speed.data.mask,
                                           this_var.data.mask)

    return this_var


#************************************************************************
def get_repeating_streak_threshold(obs_var: utils.Meteorological_Variable,
                                   config_dict: dict,
                                   wind_speed: utils.Meteorological_Variable | None = None,
                                   plots: bool = False,
                                   diagnostics: bool = False) -> None:
    """
    Use distribution to determine threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param str config_dict: configuration dictionary to store critical values
    :param MetVar wind_speed: need speeds to mask calm periods in wind_directions
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    this_var = prepare_obs_var(obs_var, wind_speed=wind_speed)

    # only process further if there is enough data (need at least 2 for a streak of repeated values!)
    if len(this_var.data.compressed()) >= 2:

        repeated_streak_lengths, _, _ = utils.prepare_data_repeating_streak(this_var.data.compressed(),
                                                                            diff=0, plots=plots,
                                                                            diagnostics=diagnostics)

        ###########
        # Approach here is to look for streaks where values are the same in value space (diff=0).
        # In Humidity, looking only for streaks (in DPD) are == 0, so have filtered into a
        #    set of locations where this criterion is met.
        # So for this test, could search in location space _or_ in value space.
        #    The latter means that in the utils.prepare_data_repeating_streak()
        #    routine the differences are 0, the former pre-identifies locations where a difference
        #    is a value (either specified as per humidity, or using first differences == 0)
        #    and hence the locational difference is 1, i.e. adjacent locations identified.
        # However, use of first differences mucks up indexing, so keeping a diff=0 approach
        # Sept 2024
        ###########

        # bin width is 1 as dealing in time index.
        # minimum bin value is 2 as this is the shortest streak possible
        threshold = utils.get_critical_values(repeated_streak_lengths, binmin=2,
                                              binwidth=1.0, plots=plots,
                                              diagnostics=diagnostics,
                                              title=this_var.name.capitalize(),
                                              xlabel="Repeating streak length")

        # write out the thresholds...
        try:
            config_dict[f"STREAK-{this_var.name}"]["Straight"] = int(threshold)
        except KeyError:
            CD_straight = {"Straight" : int(threshold)}
            config_dict[f"STREAK-{this_var.name}"] = CD_straight

    else:
        # store high value so threshold never reached
        try:
            config_dict[f"STREAK-{this_var.name}"]["Straight"] = utils.MDI
        except KeyError:
            CD_straight = {"Straight" : utils.MDI}
            config_dict[f"STREAK-{this_var.name}"] = CD_straight

    # get_repeating_streak_threshold


#************************************************************************
def get_excess_streak_threshold(obs_var: utils.Meteorological_Variable,
                                years: np.ndarray, config_dict: dict,
                                wind_speed: utils.Meteorological_Variable | None = None,
                                plots: bool = False,
                                diagnostics: bool = False) -> None:
    """
    Use distribution to determine threshold values.  Then also store in config file.

    :param MetVar obs_var: meteorological variable object
    :param array years: year for each timestamp in data
    :param str config_dict: configuration dictionary to store critical values
    :param MetVar wind_speed: need speeds to mask calm periods in wind_directions
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    this_var = prepare_obs_var(obs_var, wind_speed=wind_speed)

    proportions = np.zeros(np.unique(years).shape[0])
    for y, year in enumerate(np.unique(years)):

        locs, = np.nonzero(years == year)

        if len(this_var.data[locs].compressed()) < 2:
            continue

        # Not looking for distribution of streak lengths,
        #  but proportion of obs identified as in a streak in each calendar year.

        year_repeated_streak_lengths, _, _ = utils.prepare_data_repeating_streak(this_var.data[locs].compressed(),
                                                                        diff=0, plots=plots,
                                                                        diagnostics=diagnostics)

        # Don't want all streaks (min of 2 consecutive values), need some form
        #  of higher threshold there.

        long_enough_locs, = np.nonzero(year_repeated_streak_lengths >= MIN_STREAK_LENGTH_FOR_EXCESS_FREQUENCY)

        proportions[y] = np.sum(year_repeated_streak_lengths[long_enough_locs])/len(this_var.data[locs].compressed())

    if len(np.nonzero(proportions != 0)[0]) >= 5:
        # If at least 5 years have sufficient data that can calculate a fraction

        # bin width is 0.005 (0.5%) as dealing in fractions
        # minimum bin value is 0 as this is the lowest proportion possible
        threshold = utils.get_critical_values(proportions, binmin=0,
                                              binwidth=0.005, plots=plots,
                                              diagnostics=diagnostics,
                                              title=this_var.name.capitalize(),
                                              xlabel="Excess streak proportion")

        if threshold > 1:
            # cannot have a proportion higher than 1, so setting a value isn't sensible
            threshold = utils.MDI

        # write out the thresholds...
        try:
            config_dict[f"STREAK-{this_var.name}"]["Excess"] = threshold
        except KeyError:
            CD_straight = {"Excess" : threshold}
            config_dict[f"STREAK-{this_var.name}"] = CD_straight

    else:
        # store high value so threshold never reached
        try:
            config_dict[f"STREAK-{this_var.name}"]["Excess"] = utils.MDI
        except KeyError:
            CD_straight = {"Excess" : utils.MDI}
            config_dict[f"STREAK-{this_var.name}"] = CD_straight

    # get_excess_streak_threshold


#************************************************************************
def repeating_value(obs_var: utils.Meteorological_Variable, times: np.ndarray,
                    config_dict: dict,
                    wind_speed: utils.Meteorological_Variable | None = None,
                    plots: bool = False, diagnostics: bool = False) -> None:
    """
    AKA straight streak

    Use config file to read threshold values.  Then find streaks which exceed threshold.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_dict: configuration file to store critical values
    :param MetVar wind_speed: need speeds to mask calm periods in wind_directions
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    this_var = prepare_obs_var(obs_var, wind_speed=wind_speed)

    flags = np.array(["" for i in range(this_var.data.shape[0])])
    compressed_flags = np.array(["" for i in range(this_var.data.compressed().shape[0])])
    masked_times = np.ma.array(times, mask=obs_var.data.mask)

    # retrieve the threshold
    try:
        threshold = config_dict[f"STREAK-{this_var.name}"]["Straight"]
    except KeyError:
        # no threshold set
        get_repeating_streak_threshold(this_var, config_dict, wind_speed=wind_speed,
                                       plots=plots, diagnostics=diagnostics)
        threshold = config_dict[f"STREAK-{this_var.name}"]["Straight"]

    if threshold == utils.MDI:
        # No threshold obtainable, no need to continue the test
        return

    # Only process further if there is enough data.
    #   Need at least 2 observations to get a streak, so return if 1 or 0.
    if len(this_var.data.compressed()) < 2:
        # Escape if insufficient data in this array
        return

    repeated_streak_lengths, grouped_diffs, streaks = utils.prepare_data_repeating_streak(this_var.data.compressed(), diff=0, plots=plots, diagnostics=diagnostics)

    # above threshold
    bad, = np.nonzero(repeated_streak_lengths >= threshold)

    # flag identified streaks
    for streak in bad:
        start = int(np.sum(grouped_diffs[:streaks[streak], 1]))
        end = start + int(grouped_diffs[streaks[streak], 1]) + 1

        compressed_flags[start : end] = "K"

        if plots:
            plot_streak(masked_times, this_var.data, obs_var.units, start, end)

    # undo compression and write into original object (the one with calm periods)
    flags[this_var.data.mask == False] = compressed_flags
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Repeated streaks {this_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.nonzero(flags != '')[0])}")

    # repeating_value


#************************************************************************
def excess_repeating_value(obs_var: utils.Meteorological_Variable, times: np.ndarray,
                    config_dict: dict,
                    wind_speed: utils.Meteorological_Variable | None = None,
                    plots: bool = False, diagnostics: bool = False) -> None:
    """
    Flag years where more than expected fraction of data occurs in streaks,
      but none/not many are long enough in themselves to trigger the repeating_value check
      Themselves need to be a reasonable length (10) else this overflags

    Use config file to read threshold values.  Then find streaks which exceed threshold.

    :param MetVar obs_var: meteorological variable object
    :param array times: array of times (usually in minutes)
    :param str config_dict: configuration file to store critical values
    :param MetVar wind_speed: need speeds to mask calm periods in wind_directions
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output

    """
    years = np.array([t.year for t in times])
    if plots:
        # Needed for plotting only
        masked_times = np.ma.array(times, mask=obs_var.data.mask)

    this_var = prepare_obs_var(obs_var, wind_speed=wind_speed)

    # use mask rather than compress immediately, to ensure indexing works
    flags = np.ma.array(["" for i in range(this_var.data.shape[0])],
                        mask=this_var.data.mask)

    # retrieve the threshold and store
    try:
        threshold = config_dict[f"STREAK-{this_var.name}"]["Excess"]
    except KeyError:
        # no threshold set
        get_excess_streak_threshold(this_var, years, config_dict, wind_speed=wind_speed,
                                    plots=plots, diagnostics=diagnostics)
        threshold = config_dict[f"STREAK-{this_var.name}"]["Excess"]


    if threshold == utils.MDI:
        # No threshold obtainable, no need to continue the test
        return

    # Spin through each year, calculate proportion, flag comprising streaks if too high
    for year in np.unique(years):
        # set up indices for year, and flags array for year
        locs, = np.nonzero(years == year)
        year_flags = flags[locs]
        unmasked, = np.nonzero(year_flags.mask == False)

        if len(this_var.data[locs].compressed()) < 2:
            # Escape if insufficient data
            continue

        # Not looking for distribution of streak lengths,
        #  but proportion of obs identified as in a streak in each calendar year.
        (year_repeated_streak_lengths,
        grouped_diffs,
        streaks) = utils.prepare_data_repeating_streak(this_var.data[locs].compressed(),
                                                        diff=0, plots=plots,
                                                        diagnostics=diagnostics)

        long_enough_streaks, = np.nonzero(year_repeated_streak_lengths >= MIN_STREAK_LENGTH_FOR_EXCESS_FREQUENCY)

        proportion = np.sum(year_repeated_streak_lengths[long_enough_streaks])/len(this_var.data[locs].compressed())

        if proportion <= threshold:
            # Move on to next year
            continue

        # all identified streaks (>= MIN_STREAK_LENGTH_FOR_EXCESS_FREQUENCY in a row)
        for streak in long_enough_streaks:
            start = int(np.sum(grouped_diffs[:streaks[streak], 1]))
            end = start + int(grouped_diffs[streaks[streak], 1]) + 1

            year_flags[unmasked[start : end]] = "x"

            if plots:
                plot_streak(masked_times, this_var.data[locs], obs_var.units, start, end)

        flags[locs] = year_flags

    # Write into original object (the one with calm periods)
    obs_var.flags = utils.insert_flags(obs_var.flags, flags)

    logger.info(f"Excess Repeated streaks {this_var.name}")
    logger.info(f"   Cumulative number of flags set: {len(np.ma.nonzero(flags != '')[0])}")

    # excess_repeating_value


#************************************************************************
def repeating_day(obs_var: utils.Meteorological_Variable, station: utils.Station,
                  config_dict: dict, determine_threshold: bool = False,
                  plots: bool = False, diagnostics: bool = False) -> None:
    """
    Find and flag instances where there are streaks of days which repeat exactly.
    This is possible for certain variables in certain synoptic setups, so again
    find a threshold and flag those above.  Only looking for repeats which are
    adjacent, identical days separated by other data are not identified.

    :param MetVar obs_var: meteorological variable object
    :param Station station: Station object
    :param str config_dict: configuration file to store critical values
    :param bool determine_threshold: either calculate a threshold (True), or set flags (False)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    # any complete repeat of 24hs (as long as not "straight streak")

    set_flags = not determine_threshold

    if set_flags:
        flags = np.array(["" for _ in obs_var.data])
        # retrieve the threshold and store
        try:
            threshold = config_dict[f"STREAK-{obs_var.name}"]["DayRepeat"]
        except KeyError:
            # no threshold set
            # For this routine, can't set to call the setting routine as it is itself
            return

    # sort into array of days
    # check each day and compare to previous. HadISD used fixed threshold, but could do one pass to get, second to apply/?
    all_lengths = []
    streak_length = 0
    previous_day_data = np.array([])
    streak_locs = np.array([], dtype=(int))

    for year in np.sort(np.unique(station.years)):
        for month in np.arange(1, 13, 1):
            for day in np.arange(1, 32, 1):
                try:
                    # if Datetime doesn't throw an error, then valid date
                    _ = dt.datetime(year, month, day)
                except ValueError:
                    # not a valid date (e.g. Feb 30th, November 31st)
                    continue

                this_day_locs, = np.nonzero(np.logical_and.reduce((station.years == year,
                                                        station.months == month,
                                                        station.days == day)))

                if len(this_day_locs) == 0:
                    # skip if no data
                    continue

                this_day_data = obs_var.data[this_day_locs]

                if len(this_day_data.compressed()) == 0:
                    # skip if no unmasked data
                    continue

                if np.array_equal(this_day_data, previous_day_data):
                    # if day equals previous, part of a streak, add to counters
                    streak_length += 1
                    streak_locs = np.append(streak_locs, this_day_locs)
                    if streak_length == 1:
                        # if beginning of a new streak, count previous day as well
                        streak_length += 1
                        streak_locs = np.append(streak_locs, previous_day_locs)

                else:
                    # if different, then if end of a streak, save, set flags, and reset
                    if streak_length != 0:
                        if set_flags and streak_length > threshold:
                            # Apply the flags
                            flags[streak_locs] = "y"

                        all_lengths += [streak_length]
                        streak_length = 0
                        streak_locs = np.array([], dtype=(int))

                # make copies for next loop
                previous_day_data = np.ma.copy(this_day_data)
                previous_day_locs = np.ma.copy(this_day_locs)

    # Calculate and save the threshold.
    if determine_threshold:
        threshold = utils.get_critical_values(all_lengths, binwidth=1,
                                plots=plots,title=obs_var.name.capitalize(),
                                xlabel="Streaks of repeating days")

        try:
            config_dict[f"STREAK-{obs_var.name}"]["DayRepeat"] = threshold
        except KeyError:
            CD_dayrepeat = {"DayRepeat" : threshold}
            config_dict[f"STREAK-{obs_var.name}"] = CD_dayrepeat

    if set_flags:
        # Write into original object (the one with calm periods)
        obs_var.flags = utils.insert_flags(obs_var.flags, flags)

        logger.info(f"Repeated Day streaks {obs_var.name}")
        logger.info(f"   Cumulative number of flags set: {len(np.nonzero(flags != '')[0])}")

    # repeating_day


#************************************************************************
#def hourly_repeat():

    # repeat of given value at same hour of day for > N days
    # HadISD used fixed threshold.  Perhaps can dynamically calculate?

    # sort in to 24hr expanded array
    # for each hour
    # check each day, HadISD used fixed threshold, but could do one pass to get, second to apply/?


#    return # hourly_repeat



#************************************************************************
def rsc(station: utils.Station, var_list: list, config_dict: dict,
        full: bool = False, plots: bool = False, diagnostics: bool = False) -> None:
    """
    Run through the variables and pass to the Repeating Streak Checks

    :param Station station: Station Object for the station
    :param list var_list: list of variables to test
    :param str config_dict: dictionary for configuration settings
    :param bool full: run a full update (recalculate thresholds)
    :param bool plots: turn on plots
    :param bool diagnostics: turn on diagnostic output
    """

    for var in var_list:

        obs_var = getattr(station, var)
        if obs_var.name == "wind_direction":
            wind_speed = copy.deepcopy(getattr(station, "wind_speed"))
        else:
            wind_speed = None

        # need to have at least two observations to get a streak
        if len(obs_var.data.compressed()) < 2:
            continue

        if full:
            # Recalculating all thresholds for a full run of the QC suite

            # Get thresholds for simple repeating values
            get_repeating_streak_threshold(obs_var, config_dict, wind_speed=wind_speed,
                                            plots=plots, diagnostics=diagnostics)

            # Get thresholds for years with excess numbers of streaks
            get_excess_streak_threshold(obs_var, station.years, config_dict, wind_speed=wind_speed,
                                        plots=plots, diagnostics=diagnostics)

            # Complete repeat of a day's values
            # This is slow - as spinning through each day and then selecting
            repeating_day(obs_var, station, config_dict, determine_threshold=True,
                            plots=plots, diagnostics=diagnostics)

        # Now run the routines to apply the thresholds and set flags

        # Simple streaks of repeated values
        repeating_value(obs_var, station.times, config_dict, wind_speed=wind_speed,
                        plots=plots, diagnostics=diagnostics)

        # more short streaks than reasonable
        excess_repeating_value(obs_var, station.times, config_dict, wind_speed=wind_speed,
                                plots=plots, diagnostics=diagnostics)

        # repeats at same hour of day
        # hourly_repeat()

        # repeats of whole day
        #   (slow)
        try:
            # If there is a threshold set, then can use
            _ = config_dict[f"STREAK-{obs_var.name}"]["DayRepeat"]
        except KeyError:
            # if no threshold set, then need to run script to calculate it,
            #   even if full=False
            repeating_day(obs_var, station, config_dict, determine_threshold=True, plots=plots, diagnostics=diagnostics)
        repeating_day(obs_var, station, config_dict, determine_threshold=False, plots=plots, diagnostics=diagnostics)

    # rsc

#************************************************************************
if __name__ == "__main__":

    print("checking repeated streaks")
#************************************************************************
