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
import numpy as np
import datetime as dt
import pandas as pd

# internal utils
import qc_utils as utils
import io_utils as io
import qc_tests
#************************************************************************

# TODO - Sphinx


# Temporary stuff
IFF_LOC = "/data/users/rdunn/Copernicus/c3s311a_lot2/iff"

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

    # Need codes to process IDs and any inventory
    station_list = ["WMO02474-1_220.psv"]
    obs_var_list = ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure", "wind_direction", "wind_speed"]

    for st, station_id in enumerate(station_list):

        # config file?
        config_file = "{}.config".format(station_id)


        # set up the stations
        # TODO - read in a station list correctly!
        lat = 52
        lon = 0.1
        elev = 10
        station = utils.Station(station_id, lat, lon, elev)

        #*************************
        # read MFF 
        station_df = io.read(os.path.join(IFF_LOC, station_id[:-4]))
        # convert to datetimes
        datetimes = pd.to_datetime(station_df[["Year", "Month", "Day", "Hour", "Minute"]])

        # convert dataframe to station and MetVar objects for internal processing
        utils.populate_station(station, station_df, obs_var_list)
        station.times = datetimes
        
        # store extra information to enable easy extraction later
        station.years = station_df["Year"].fillna(utils.MDI).to_numpy()
        station.months = station_df["Month"].fillna(utils.MDI).to_numpy()
        station.hours = station_df["Hour"].fillna(utils.MDI).to_numpy()

        #*************************
        # lat and lon checks

        """
        HadISD tests and order

        Duplicated months
        Odd Clusters of data
        Frequent Values - tick
        Diurnal Cycle
        Gaps in distributions
        World Records - tick
        Repeated values (streaks or just too common short ones) - partial tick
        Climatology
        Spike - tick
        Humidity Cross checks - super saturation, dewpoint depression, dewpoint cut off - tick
        Cloud logical checks
        Excess Variance
        Winds (logical wind & wind rose)
        Logical SLP/StnLP - tick
        Precipitation logical checks
        """
        #*************************

        # TODO - sort updating vs not of config files
        # TODO - use suite config file to store all settings for tests

#        qc_tests.streaks.rsc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, plots=plots, diagnostics=diagnostics)

#        qc_tests.spike.sc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, plots=plots, diagnostics=diagnostics)

#        qc_tests.world_records.wrc(station, ["temperature", "dew_point_temperature", "sea_level_pressure", "wind_speed"], plots=plots, diagnostics=diagnostics)

#        qc_tests.humidity.hcc(station, plots=plots, diagnostics=diagnostics)

#        qc_tests.frequent.fvc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, plots=plots, diagnostics=diagnostics)

#        qc_tests.pressure.pcc(station, config_file, plots=plots, diagnostics=diagnostics)

#        station.temperature.data[::100] += 40

#        qc_tests.distribution.dgc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, plots=plots, diagnostics=diagnostics)

        # not run on pressure data in HadISD.
#        qc_tests.climatological.coc(station, ["temperature", "dew_point_temperature"], config_file, plots=plots, diagnostics=diagnostics)

#        station.temperature.data[-200:] *= 2
#        qc_tests.variance.evc(station, ["temperature", "dew_point_temperature", "station_level_pressure", "sea_level_pressure"], config_file, plots=plots, diagnostics=diagnostics)

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
        io.write(os.path.join(IFF_LOC, "{}_QC".format(station_id[:-4])), station_df)

        input("end")

    return # run_checks

#************************************************************************
if __name__=="__main__":

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
