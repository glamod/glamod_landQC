'''
Inter station checks (between station records)
==============================================

This script calls the individual QC check routines for the buddy checks on each station in turn.
You can pass a single station by setting the ``--restart_id`` and ``-end_id`` to
equal the same station ID, do a range by giving different IDs, or run all
stations in the station list by leaving empty.

inter_checks.py invoked by typing::

  python inter_checks.py --restart_id --end_id [--full] [--plots] [--diagnostics] [--test] [--clobber]

with an example call being of the form::

  python inter_checks.py --restart_id AAI0000TNCA --end_id AAI0000TNCA --full --clobber

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

--plots             [False] Create plots (maybe interactive)

--diagnostics       [False] Verbose output

--test              ["all"] select a single test to run [neighbour/clean_up/high_flag]

--clobber           Overwrite output files if already existing.  If not set, will skip if output exists

'''
#************************************************************************
import os
import datetime as dt
import numpy as np
import logging

# internal utils
import qc_utils as utils
import io_utils as io
import qc_tests
import setup
#************************************************************************

#************************************************************************
def read_neighbours(restart_id: str = "", end_id: str = "") -> np.ndarray:
    """
    Read the neighbour file to store neighbours and distances [station, neighbours, distances]

    :param str restart_id: which station to start on
    :param str end_id: which station to end on

    :returns: array - [station, neighbours, distances]
    """

    all_entries = np.genfromtxt(os.path.join(setup.SUBDAILY_CONFIG_DIR, utils.NEIGHBOUR_FILE), dtype=(str))
    station_IDs = all_entries[:, 0]

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = np.where(all_entries[:, 0] == end_id)
        all_entries = all_entries[: endindex[0]+1]

    # and do the front
    if restart_id != "":
        startindex, = np.where(station_IDs == restart_id)
        all_entries = all_entries[startindex[0]:]

    all_entries = all_entries.reshape((all_entries.shape[0], utils.MAX_N_NEIGHBOURS, 2))

    return all_entries # read_neighbours

#************************************************************************
def run_checks(restart_id:str = "", end_id:str = "", diagnostics:bool = False, plots: bool = False,
               full: bool = False, test: str = "all", clobber: bool = False) -> None:
    """
    Main script.  Reads in station data, populates internal objects and passes to the tests.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    :param str test: specify a single test to run (useful for diagnostics) [neighbour/clean_up/high_flag]
    :param bool clobbber: overwrite output file if exists
    """

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)
    station_IDs = station_list.id

    # read in all the neighbours for these stations to hold ready
    all_neighbours = read_neighbours(restart_id=restart_id, end_id=end_id)

    # now spin through each ID in the curtailed list
    for st, target_station_id in enumerate(station_IDs):
        print(f"{dt.datetime.now()} {target_station_id} ({st+1}/{station_IDs.shape[0]})")

        if not clobber:
            # wanting to skip if files exist
            if os.path.exists(os.path.join(setup.SUBDAILY_BAD_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}")):
                print(os.path.join(setup.SUBDAILY_BAD_DIR, "f{target_station_id:11s}.qff{setup.OUT_COMPRESSION}") +
                      "exists and clobber kwarg not set, skipping to next station.")
                continue
            elif os.path.exists(os.path.join(setup.SUBDAILY_OUT_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}")):
                print(os.path.join(setup.SUBDAILY_OUT_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}") +
                      "exists and clobber kwarg not set, skipping to next station.")
                continue
            else:
                # files don't exists, pass
                pass
        else:
            if diagnostics: print(f"Overwriting output for {target_station_id}")
        startT = dt.datetime.now()

        #*************************
        # set up logging
        logfile = os.path.join(setup.SUBDAILY_LOG_DIR, f"{target_station_id}_external_checks.log")
        if os.path.exists(logfile):
            os.remove(logfile)
        logger = utils.custom_logger(logfile)
        logger.info(f"External (Buddy) Checks on {target_station_id}")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        #*************************
        # set up the stations
        target_station = utils.Station(target_station_id, station_list.latitude[st], station_list.longitude[st], station_list.elevation[st])
        if diagnostics:
            print(target_station)

        try:
            target_station, target_station_df = io.read_station(os.path.join(
                setup.SUBDAILY_PROC_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}"),
                                                                target_station, read_flags=True)
        except FileNotFoundError:
            # file missing, move on to next in sequence
            logging.warning(f"File for {target_station.id} missing")
            print("") # for on screen spacing of text
            continue

        # some may have no data (for whatever reason)
        if target_station.times.shape[0] == 0:
            logging.warning(f"No data in station {target_station.id}")
            if diagnostics:
                print("No data in station {target_station.id}")
            # scoot onto next station
            print("")
            continue

        # extract neighbours for this station
        nloc, = np.where(all_neighbours[:, 0, 0] == target_station_id)
        initial_neighbours = all_neighbours[nloc].squeeze()

        #*************************
        # TODO: refine neighbours [quadrants, correlation?]

        if test in ["all", "outlier"]:
            if diagnostics: print("N", dt.datetime.now()-startT)
            qc_tests.neighbour_outlier.noc(target_station, initial_neighbours, \
                                               ["temperature", "dew_point_temperature", "wind_speed", "station_level_pressure", "sea_level_pressure"], full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "clean_up"]:
            if diagnostics: print("U", dt.datetime.now()-startT)
            qc_tests.clean_up.mcu(target_station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed", "wind_direction"], full=full, plots=plots, diagnostics=diagnostics)


        if test in ["all", "high_flag"]:
            if diagnostics: print("H", dt.datetime.now()-startT)
            hfr_vars_set = qc_tests.high_flag.hfr(target_station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed", "wind_direction"], full=full, plots=plots, diagnostics=diagnostics)

        # write in the flag information
        for var in setup.obs_var_list:
            obs_var = getattr(target_station, var)
            target_station_df[f"{var}_QC_flag"] = obs_var.flags

        #*************************
        # Output of QFF
        # write out the dataframe to output format
        if hfr_vars_set > 1:
            # high flagging rates in more than one variable.  Withholding station completely
            if diagnostics: print(f"{target_station.id} withheld as too high flagging")
            logging.info(f"{target_station.id} withheld as too high flagging")
            io.write(os.path.join(setup.SUBDAILY_BAD_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}"),
                     target_station_df, formatters={"Latitude" : "{:7.4f}", "Longitude" : "{:7.4f}", "Month": "{:02d}", "Day": "{:02d}", "Hour" : "{:02d}", "Minute" : "{:02d}"})

        else:
            io.write(os.path.join(setup.SUBDAILY_OUT_DIR, f"{target_station_id:11s}.qff{setup.OUT_COMPRESSION}"),
                     target_station_df, formatters={"Latitude" : "{:7.4f}", "Longitude" : "{:7.4f}", "Month": "{:02d}", "Day": "{:02d}", "Hour" : "{:02d}", "Minute" : "{:02d}"})


        #*************************
        # Output flagging summary file
        io.flag_write(os.path.join(setup.SUBDAILY_FLAG_DIR, f"{target_station_id:11s}.flg"), target_station_df, diagnostics=diagnostics)

        if diagnostics or plots:
            input(f"Stop after {dt.datetime.now()-startT} of processing")
            return

    return # run_checks

#************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart_id', dest='restart_id', action='store', default="",
                        help='Restart ID for truncated run, default=""')
    parser.add_argument('--end_id', dest='end_id', action='store', default="",
                        help='End ID for truncated run, default=""')
    parser.add_argument('--full', dest='full', action='store_true', default=False,
                        help='Run full reprocessing rather than just an update')
    parser.add_argument('--diagnostics', dest='diagnostics', action='store_true', default=False,
                        help='Run diagnostics (will not write out file)')
    parser.add_argument('--plots', dest='plots', action='store_true', default=False,
                        help='Run plots (will not write out file)')
    parser.add_argument('--test', dest='test', action='store', default="all",
                        help='Select single test [neighbour/clean_up/high_flag]')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False,
                        help='Overwrite output files if they exists.')


    args = parser.parse_args()

    run_checks(restart_id=args.restart_id,
               end_id=args.end_id,
               diagnostics=args.diagnostics,
               plots=args.plots,
               full=args.full,
               test=args.test,
               clobber=args.clobber,
           )

#************************************************************************
