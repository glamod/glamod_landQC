'''
Intra station checks (within station record)

intra_checks.py invoked by typing::

  python intra_checks.py --restart_id --end_id [--full] [--plots] [--diagnostics] [--test]

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

--plots             [False] Create plots (maybe interactive)

--diagnostics       [False] Verbose output

--test              ["all"] select a single test to run [climatological/distribution/diurnal
                     frequent/humidity/odd_cluster/pressure/spike/streaks/timestamp/variance/winds/world_records]
'''
#************************************************************************
import os
import datetime as dt
import numpy as np
import pandas as pd

# internal utils
import qc_utils as utils
import io_utils as io
import qc_tests
import setup
#************************************************************************

#************************************************************************
def run_checks(restart_id="", end_id="", diagnostics=False, plots=False, full=False, test="all"):
    """
    Main script.  Reads in station data, populates internal objects and passes to the tests.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    :param str test: specify a single test to run (useful for diagnostics) [climatological/distribution/diurnal
                     frequent/humidity/odd_cluster/pressure/spike/streaks/timestamp/variance/winds/world_records]
    """

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    station_IDs = station_list.id

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

        startT = dt.datetime.now()
        # set up config file to hold thresholds etc
        config_file = os.path.join(setup.SUBDAILY_CONFIG_DIR, "{:11s}.config".format(station_id))
        if full:
            try:
                # recreating, so remove completely
                os.remove(config_file)
            except IOError:
                pass
    

        #*************************
        # set up the stations
        station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st], station_list.elevation[st])
        if diagnostics:
            print(station)

        try:
            station, station_df = io.read_station(os.path.join(setup.SUBDAILY_MFF_DIR, "{:11s}.mff".format(station_id)), station)
        except OSError as e:
            # file missing, move on to next in sequence
            io.write_error(station, "File Missing")
            continue
        except ValueError as e:
            # some issue in the raw file
            io.write_error(station, "Error in input file", error=str(e))
            continue

        # some may have no data (for whatever reason)
        if station.times.shape[0] == 0:
            if diagnostics:
                print("No data in station {}".format(station.id))
            # scoot onto next station
            io.write_error(station, "No data in input file")
            continue

        #*************************

        """
        HadISD tests and order

        Duplicated months
        Odd Clusters of data - need to address output with buddy checks in due course.
        Frequent Values - tick
        Diurnal Cycle
        Gaps in distributions - tick
        World Records - tick
        Repeated values (streaks or just too common short ones) - partial tick
        Climatology - tick
        Spike - tick
        Humidity Cross checks - super saturation, dewpoint depression, dewpoint cut off - tick (dewpoint cut off not applied)
        Cloud logical checks - clouds not in C3S 311a @Aug 2019
        Excess Variance - partial tick
        Winds (logical wind & wind rose) - logical tick.  Not sure if wind rose is robust enough
        Logical SLP/StnLP - tick
        Precipitation logical checks - precip not in C3S 311a @Aug 2019
        """
        #*************************
        if test in ["all", "logic"]:
            # incl lat, lon and elev checks
#
            print("L", dt.datetime.now()-startT)
            good_metadata = qc_tests.logic_checks.lc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed", "wind_direction"], full=full, plots=plots, diagnostics=diagnostics)

            if good_metadata != 0:
                print("Issue with station metadata")
                # skip on to next one
                continue

        if test in ["all", "odd_cluster"]:
            print("O", dt.datetime.now()-startT)
            # TODO - use suite config file to store all settings for tests
            qc_tests.odd_cluster.occ(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "frequent"]:
            print("F", dt.datetime.now()-startT)
            qc_tests.frequent.fvc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        # HadISD only runs on stations where latitude lower than 60(N/S)
        # Takes a long time, this one
        if test in ["all", "diurnal"]:
            print("U", dt.datetime.now()-startT)
            if np.abs(station.lat < 60):
                qc_tests.diurnal.dcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "distribution"]:
            print("D", dt.datetime.now()-startT)
            qc_tests.distribution.dgc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "world_records"]:
            print("W", dt.datetime.now()-startT)
            qc_tests.world_records.wrc(station, ["temperature", "dew_point_temperature", "sea_level_pressure", "wind_speed"], full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "streaks"]:
            print("K", dt.datetime.now()-startT)
            qc_tests.streaks.rsc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed", "wind_direction"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        # not run on pressure data in HadISD.
        if test in ["all", "climatological"]:
            print("C", dt.datetime.now()-startT)
            qc_tests.climatological.coc(station, ["temperature", "dew_point_temperature"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "timestamp"]:
            print("T", dt.datetime.now()-startT)
            qc_tests.timestamp.tsc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "spike"]:
            print("S", dt.datetime.now()-startT)
            qc_tests.spike.sc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "humidity"]:
            print("h", dt.datetime.now()-startT)
            qc_tests.humidity.hcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "variance"]:
            print("V", dt.datetime.now()-startT)
            qc_tests.variance.evc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "pressure"]:
            print("P", dt.datetime.now()-startT)
            qc_tests.pressure.pcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        if test in ["all", "winds"]:
            print("w", dt.datetime.now()-startT)
            qc_tests.winds.wcc(station, config_file, fix=True, full=full, plots=plots, diagnostics=diagnostics)


        print(dt.datetime.now()-startT)

        #*************************
        # Insert flags into Data Frame

        # need to insert columns in correct place
        column_names = station_df.columns.values

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

        # # sort source_ID.x columns - purely for first release
        # for c, column in enumerate(station_df.columns):
        #     if "Source_ID" in column:
        #         # replace the NaN with empty string
        #         station_df[column] = station_df[column].fillna('')
        #         # rename the column
        #         variable = station_df.columns[c-1]
        #         station_df = station_df.rename(columns={column : "{}_Source_ID".format(variable)})
                

        # write in the flag information
        for var in setup.obs_var_list:
            obs_var = getattr(station, var)
            station_df["{}_QC_flag".format(var)] = obs_var.flags
       

        #*************************
        # Output of QFF
        # write out the dataframe to output format
        if hfr_vars_set > 1:
            # high flagging rates in more than one variable.  Withholding station completely
            print("{} withheld as too high flagging".format(station.id))
            io.write(os.path.join(setup.SUBDAILY_BAD_DIR, "{:11s}.qff".format(station_id)), station_df)
        else:
            io.write(os.path.join(setup.SUBDAILY_PROC_DIR, "{:11s}.qff".format(station_id)), station_df)

        #*************************
        # Output flagging summary file
        io.flag_write(os.path.join(setup.SUBDAILY_FLAG_DIR, "{:11s}.flg".format(station_id)), station_df, diagnostics=diagnostics)


        print(dt.datetime.now()-startT)

#        if diagnostics or plots:
#            input("end")
#            break

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
    args = parser.parse_args()

    run_checks(restart_id=args.restart_id,
               end_id=args.end_id,
               diagnostics=args.diagnostics,
               plots=args.plots,
               full=args.full,
               test=args.test)

#************************************************************************
