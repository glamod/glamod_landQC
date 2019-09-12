'''
Intra station checks (within station record)

intra_checks.py invoked by typing::

  python intra_checks.py --restart_id --end_id [--full] [--plots] [--diagnostics]

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

--plots             [False] Create plots (maybe interactive)

--diagnostics       [False] Verbose output
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

# Temporary stuff
MFF_LOC = setup.SUBDAILY_IN_DIR
QFF_LOC = setup.SUBDAILY_OUT_DIR
CONF_LOC = setup.SUBDAILY_CONFIG_DIR


#************************************************************************
def run_checks(restart_id="", end_id="", diagnostics=False, plots=False, full=False):
    """
    Main script.  Reads in station data, populates internal objects and passes to the tests.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots from each test
    :param bool full: run full reprocessing rather than using stored values.
    """

    obs_var_list = setup.obs_var_list

    # process the station list
    station_list = pd.read_fwf(os.path.join(setup.SUBDAILY_IN_DIR, "ghcnh-stations.txt"), widths=(11, 9, 10, 7, 35), header=None)
    station_IDs = station_list.iloc[:, 0]

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = np.where(station_IDs == end_id)
        station_list = station_list.iloc[: endindex[0]+1]
        station_IDs = station_IDs[: endindex[0]+1]

    # and do the front
    if restart_id != "":
        startindex, = np.where(station_IDs == restart_id)
        station_list = station_list.iloc[startindex[0]:]
        station_IDs = station_IDs[startindex[0] :]

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print("{} {}".format(dt.datetime.now(), station_id))

        # for diagnostics
#        if station_id != "ICAOKAYE-1_223.psv": continue

        startT = dt.datetime.now()
        # set up config file to hold thresholds etc
        config_file = os.path.join(CONF_LOC, "{}.config".format(station_id))

        #*************************
        # set up the stations
        # TEMPORARY
        # extract geo metadata from DF
#        station = utils.Station(station_id, station_df["Latitude"][0], station_df["Longitude"][0], station_df["Elevation"][0])

        station = utils.Station(station_id, station_list.iloc[st][1], station_list.iloc[st][2], station_list.iloc[st][3])
        if diagnostics:
            print(station)

        #*************************
        # read MFF
        station_df = io.read(os.path.join(MFF_LOC, station_id))


        # convert to datetimes
        datetimes = pd.to_datetime(station_df[["Year", "Month", "Day", "Hour", "Minute"]])

        # convert dataframe to station and MetVar objects for internal processing
        utils.populate_station(station, station_df, obs_var_list)
        station.times = datetimes

        # store extra information to enable easy extraction later
        station.years = station_df["Year"].fillna(utils.MDI).to_numpy()
        station.months = station_df["Month"].fillna(utils.MDI).to_numpy()
        station.days = station_df["Day"].fillna(utils.MDI).to_numpy()
        station.hours = station_df["Hour"].fillna(utils.MDI).to_numpy()

        #*************************
        # lat and lon checks

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
        print("O", dt.datetime.now()-startT)
        # TODO - use suite config file to store all settings for tests
        qc_tests.odd_cluster.occ(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("F", dt.datetime.now()-startT)
        qc_tests.frequent.fvc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        # HadISD only runs on stations where latitude higher than 60(N/S)
        # Takes a long time, this one
#        print("U", dt.datetime.now()-startT)
#        if np.abs(station.latitude < 60):
#            qc_tests.diurnal.dcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("D", dt.datetime.now()-startT)
        qc_tests.distribution.dgc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("W", dt.datetime.now()-startT)
        qc_tests.world_records.wrc(station, ["temperature", "dew_point_temperature", "sea_level_pressure", "wind_speed"], full=full, plots=plots, diagnostics=diagnostics)

        # could run on wind direction?
        print("K", dt.datetime.now()-startT)
        qc_tests.streaks.rsc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        # not run on pressure data in HadISD.
        print("C", dt.datetime.now()-startT)
        qc_tests.climatological.coc(station, ["temperature", "dew_point_temperature"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("T", dt.datetime.now()-startT)
        qc_tests.timestamp.tsc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("S", dt.datetime.now()-startT)
        qc_tests.spike.sc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("H", dt.datetime.now()-startT)
        qc_tests.humidity.hcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("V", dt.datetime.now()-startT)
        qc_tests.variance.evc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_speed"], config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("P", dt.datetime.now()-startT)
        qc_tests.pressure.pcc(station, config_file, full=full, plots=plots, diagnostics=diagnostics)

        print("w", dt.datetime.now()-startT)
        qc_tests.winds.wcc(station, config_file, fix=True, full=full, plots=plots, diagnostics=diagnostics)

        print(dt.datetime.now()-startT)

        #*************************
        # Output of QFF

        # need to insert columns in correct place
        column_names = station_df.columns.values

        # add QC flag columns to each variable
        #    initialise with blank
        #    need to automate the column identification
        new_column_indices = []
        for c, column in enumerate(station_df.columns):
            if column in obs_var_list:
                new_column_indices += [c + 2] # 2 offset rightwards from variable's column

        # reverse order so can insert without messing up the indices
        new_column_indices.reverse()
        for index in new_column_indices:
            station_df.insert(index, "{} QC flags".format(station_df.columns[index-2]), ["" for i in range(station_df.shape[0])], True)

        # write in the flag information
        for var in obs_var_list:
            obs_var = getattr(station, var)
            station_df["{} QC flags".format(var)] = obs_var.flags

        # write out the dataframe to output format
        io.write(os.path.join(QFF_LOC, "{}".format(station_id)), station_df)

        print(dt.datetime.now()-startT)

        if diagnostics or plots:
#            input("end")
            break
        
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
    args = parser.parse_args()

    run_checks(restart_id=args.restart_id,
               end_id=args.end_id,
               diagnostics=args.diagnostics,
               plots=args.plots,
               full=args.full)

#************************************************************************
