'''
Intra station checks (within station record)
============================================

This script calls the individual QC check routines on each station in turn.
You can pass a single station by setting the ``--restart_id`` and ``-end_id`` to
equal the same station ID, do a range by giving different IDs, or run all
stations in the station list by leaving empty.

intra_checks.py invoked by typing::

  python intra_checks.py --restart_id --end_id [--full] [--plots] [--diagnostics] [--test] [--clobber]

with an example call being of the form::

  python intra_checks.py --restart_id AAI0000TNCA --end_id AAI0000TNCA --full --clobber

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

--plots             [False] Create plots (maybe interactive)

--diagnostics       [False] Verbose output

--test              ["all"] select a single test to run [climatological/distribution/diurnal
                     frequent/humidity/odd_cluster/pressure/spike/streaks/timestamp/variance/winds/world_records/precision]

--clobber           Overwrite output files if already existing.  If not set, will skip if output exists

.. note::

    When selecting ``--plots`` option, do take care, as each selected QC-check will show every plot
    that the routine is set up to do.  You may want to select a single test or do some further editing
    to ensure that the features and functionality you're trying to investigate are shown easily.

'''
#************************************************************************
import datetime as dt
import numpy as np
import logging
import json
from json.decoder import JSONDecodeError

# internal utils
import utils
import io_utils as io
import qc_tests
import setup

#************************************************************************
def run_checks(restart_id: str = "", end_id: str = "", diagnostics: bool = False, plots: bool = False,
               full: bool = False, test: str = "all", clobber: bool = False) -> None:
    """
    Main script.  Reads in station data, populates internal objects and passes to the tests.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    :param str test: specify a single test to run (useful for diagnostics) [climatological/distribution/diurnal
                     frequent/humidity/odd_cluster/pressure/spike/streaks/timestamp/variance/winds/world_records/precision]
    :param bool clobbber: overwrite output file if exists
    """

    if test not in ["all", "logic", "climatological", "distribution", "diurnal", "frequent",
                    "humidity", "odd_cluster", "pressure", "spike", "streaks", "high_flag",
                    "timestamp", "variance", "winds" ,"world_records", "precision"]:
        print("Invalid test selected")
        return

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    station_IDs = station_list.id

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

        if not clobber:
            # wanting to skip if files exist
            bad_file = setup.SUBDAILY_BAD_DIR / "{:11s}{}{}".format(station_id,
                                                                    setup.OUT_SUFFIX,
                                                                    setup.OUT_COMPRESSION)
            good_file = setup.SUBDAILY_PROC_DIR / "{:11s}{}{}".format(station_id,
                                                                 setup.OUT_SUFFIX,
                                                                 setup.OUT_COMPRESSION)
            if bad_file.exists():
                print(f"{bad_file} exists and clobber kwarg not set, skipping to next station.")
                continue
            elif good_file.exists():
                print(f"{good_file} exists and clobber kwarg not set, skipping to next station.")
                continue
            else:
                # files don't exists, pass
                pass
        else:
            if diagnostics: print(f"Overwriting output for {station_id}")
        startT = dt.datetime.now()

        #*************************
        # set up logging
        logfile = setup.SUBDAILY_LOG_DIR / f"{station_id}_internal_checks.log"
        if logfile.exists():
            logfile.unlink()
        logger = utils.custom_logger(logfile)
        logger.info(f"Internal Checks on {station_id}")
        logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        #*************************
        # set up & store config file to hold thresholds etc
        config_file_name = setup.SUBDAILY_CONFIG_DIR / "{:11s}.json".format(station_id)
        if full:
            try:
                # recreating, so remove completely
                config_file_name.unlink()
                # JSON stores in dictionary, so create empty one
                config_dict = {}
            except FileNotFoundError:
                config_dict = {}
        else:
            try:
                with open(config_file_name , "r") as cfile:
                    config_dict = json.load(cfile)
            except FileNotFoundError:
                config_dict = {}
            except JSONDecodeError:
                # empty file
                print("STOP - JSON error")
                return


        #*************************
        # set up the stations
        station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st], station_list.elevation[st])
        if diagnostics:
            print(station)

        try:
            station, station_df = io.read_station(setup.SUBDAILY_MFF_DIR /
                                                  "{:11s}{}{}".format(station_id,
                                                                      setup.IN_SUFFIX,
                                                                      setup.IN_COMPRESSION), station)
        except FileNotFoundError: # as e:
            # file missing, move on to next in sequence
            io.write_error(station, "File Missing", stage="int")
            print("") # for on screen spacing of text
            continue
        except ValueError as e:
            # some issue in the raw file
            io.write_error(station, "Error in input file", error=str(e), stage="int")
            print("")
            continue
        except RuntimeError as e:
            # missing header in the raw file
            io.write_error(station, "Error in input file - missing header", error=str(e), stage="int")
            print("")
            continue

        # some may have no data (for whatever reason)
        if station.times.shape[0] == 0:
            io.write_error(station, "No data in input file", stage="int")
            logging.warning(f"No data in input file for {station.id}")
            # and scoot onto next station
            print("")
            continue

        #*************************
        # Add the country and continent
        station.country = utils.find_country_code(station.lat, station.lon)
        station.continent = utils.find_continent(station.country)


        #*************************
        if test in ["all", "logic"]:
            # incl lat, lon and elev checks

            if diagnostics: print("L", dt.datetime.now()-startT)
            good_metadata = qc_tests.logic_checks.lc(station, ["temperature",
                                                               "dew_point_temperature",
                                                               "station_level_pressure",
                                                                "sea_level_pressure",
                                                                "wind_speed",
                                                                "wind_direction"],
                                                     full=full, plots=plots, diagnostics=diagnostics)

            if good_metadata != 0:
                logging.warning("Issue with station metadata")
                # skip on to next one
                continue

        if test in ["all", "odd_cluster"]:
            if diagnostics: print("O", dt.datetime.now()-startT)
            # TODO - use suite config file to store all settings for tests
            qc_tests.odd_cluster.occ(station, ["temperature", "dew_point_temperature",
                                               "station_level_pressure", "sea_level_pressure",
                                               "wind_speed"],
                                     config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "frequent"]:
            if diagnostics: print("F", dt.datetime.now()-startT)
            qc_tests.frequent.fvc(station, ["temperature", "dew_point_temperature",
                                            "station_level_pressure", "sea_level_pressure"],
                                  config_dict, full=full, plots=plots, diagnostics=diagnostics)

        # HadISD only runs on stations where latitude lower than 60(N/S)
        # Takes a long time, this one
        if test in ["all", "diurnal"]:
            if diagnostics: print("U", dt.datetime.now()-startT)
            if np.abs(station.lat < 60):
                qc_tests.diurnal.dcc(station, config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "distribution"]:
            if diagnostics: print("D", dt.datetime.now()-startT)
            qc_tests.distribution.dgc(station, ["temperature", "dew_point_temperature",
                                                "station_level_pressure", "sea_level_pressure"],
                                      config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "world_records"]:
            if diagnostics: print("W", dt.datetime.now()-startT)
            qc_tests.world_records.wrc(station, ["temperature", "dew_point_temperature",
                                                 "sea_level_pressure", "wind_speed"],
                                       full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "streaks"]:
            if diagnostics: print("K", dt.datetime.now()-startT)
            qc_tests.streaks.rsc(station, ["temperature", "dew_point_temperature", "station_level_pressure",
                                           "sea_level_pressure", "wind_speed", "wind_direction"],
                                 config_dict, full=full, plots=plots, diagnostics=diagnostics)

        # not run on pressure data in HadISD.
        if test in ["all", "climatological"]:
            if diagnostics: print("C", dt.datetime.now()-startT)
            qc_tests.climatological.coc(station, ["temperature", "dew_point_temperature"],
                                        config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "timestamp"]:
            if diagnostics: print("T", dt.datetime.now()-startT)
            qc_tests.timestamp.tsc(station, ["temperature", "dew_point_temperature", "station_level_pressure",
                                             "sea_level_pressure", "wind_speed"],
                                   config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "precision"]:
            if diagnostics: print("n", dt.datetime.now()-startT)
            qc_tests.precision.pcc(station, config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "spike"]:
            if diagnostics: print("S", dt.datetime.now()-startT)
            qc_tests.spike.sc(station, ["temperature", "dew_point_temperature", "station_level_pressure",
                                        "sea_level_pressure", "wind_speed"],
                              config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "humidity"]:
            if diagnostics: print("h", dt.datetime.now()-startT)
            qc_tests.humidity.hcc(station, config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "variance"]:
            if diagnostics: print("V", dt.datetime.now()-startT)
            qc_tests.variance.evc(station, ["temperature", "dew_point_temperature",
                                            "station_level_pressure", "sea_level_pressure", "wind_speed"],
                                  config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "pressure"]:
            if diagnostics: print("P", dt.datetime.now()-startT)
            qc_tests.pressure.pcc(station, config_dict, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "winds"]:
            if diagnostics: print("w", dt.datetime.now()-startT)
            fixed_locs = qc_tests.winds.wcc(station, config_dict, fix=setup.FIX_WINDDIR, full=full,
                                            plots=plots, diagnostics=diagnostics)

            # fix only applies to obs_var within station, not to dataframe, hence needing to copy
            if setup.FIX_WINDDIR and len(fixed_locs) > 0:
                # take copy so can revert missing and other details
                wind_dir = np.copy(getattr(station, "wind_direction").data)
                qc_tests.qc_utils.update_dataframe(station_df, wind_dir, fixed_locs, "wind_direction")


        if test in ["all", "high_flag"]:
            if diagnostics: print("H", dt.datetime.now()-startT)
            hfr_vars_set = qc_tests.high_flag.hfr(station, ["temperature", "dew_point_temperature",
                                                            "station_level_pressure", "sea_level_pressure",
                                                            "wind_speed", "wind_direction"],
                                                  full=full, plots=plots, diagnostics=diagnostics)
        else:
            hfr_vars_set = 0

        print(f" QC checks complete in: {dt.datetime.now()-startT}")

        #*************************
        # Save the config (overwriting)
        with open(config_file_name , "w") as cfile:
            json.dump(config_dict, cfile, indent=2)


        #*************************
        # Insert flags into Data Frame

        # need to insert columns in correct place

        #*************************
        # add QC flag columns to each variable
        #    initialise with blank
        #    need to automate the column identification
        new_column_indices = []
        for c, column in enumerate(station_df.columns):
            if column in setup.obs_var_list:
                new_column_indices += [c + 2] # 2 offset rightwards from variable's column

        # reverse order so can insert without messing up the indices
        new_column_indices.reverse()
        for index in new_column_indices:
            station_df.insert(index, "{}_QC_flag".format(station_df.columns[index-2]), ["" for i in range(station_df.shape[0])], True)


        # write in the flag information
        for var in setup.obs_var_list:
            obs_var = getattr(station, var)
            station_df["{}_QC_flag".format(var)] = obs_var.flags


        #*************************
        # Output of QFF
        # write out the dataframe to output format
        if hfr_vars_set > 1:
            # high flagging rates in more than one variable.  Withholding station completely
            logging.info(f"{station.id} withheld as too high flagging")
            io.write(setup.SUBDAILY_BAD_DIR /
                    "{:11s}{}{}".format(station_id,
                                        setup.OUT_SUFFIX,
                                        setup.OUT_COMPRESSION),
                     station_df)
        else:
            io.write(setup.SUBDAILY_PROC_DIR /
                    "{:11s}{}{}".format(station_id,
                                        setup.OUT_SUFFIX,
                                        setup.OUT_COMPRESSION),
                     station_df)

        #*************************
        # Output flagging summary file
        io.flag_write(setup.SUBDAILY_FLAG_DIR / "{:11s}.flg".format(station_id),
                      station_df, diagnostics=diagnostics)
        print(" Files written\n")

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
                        help='Select single test [climatological/distribution/diurnal/frequent/humidity/odd_cluster/pressure/spike/streaks/timestamp/variance/winds/world_records]')
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
