#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import pandas as pd
import numpy as np
import setup

from qc_utils import populate_station, MDI, QC_TESTS

#************************************************************************
def read_psv(infile, separator, compression="infer"):
    '''

    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str infile: location and name of infile (without extension)
    :param str separator: separating character (e.g. ",", "|")
    :param str extension: infile extension [mff]
    :returns: df - DataFrame
    '''
    df = pd.read_csv(infile, sep=separator, compression=compression, dtype=setup.DTYPE_DICT)

    return df #  read_psv

#************************************************************************
def read(infile):
    """
    Wrapper for read functions to allow remainder to be file format agnostic.

    :param str infile: location and name of infile (without extension)
    :param str extension: infile extension [mff]
    :returns: df - DataFrame
    """

    # for .psv
    if os.path.exists(infile):
        df = read_psv(infile, "|")
    else:
        raise OSError

    return df # read

#************************************************************************
def read_station(stationfile, station, read_flags=False):
    """
    Read station info, and populate with data.

    :param str stationfile: full path to station file
    :param station station: station object with locational metadata only
    :param bool read_flags: incorporate any pre-existing flags

    :returns: station & station_df
    """
   
    #*************************
    # read MFF
    try:
        station_df = read(stationfile)
    except OSError:
        print("Missing station {}".format(stationfile))
        raise


    # convert to datetimes
    datetimes = pd.to_datetime(station_df[["Year", "Month", "Day", "Hour", "Minute"]])

    # convert dataframe to station and MetVar objects for internal processing
    populate_station(station, station_df, setup.obs_var_list, read_flags=read_flags)
    station.times = datetimes

    # store extra information to enable easy extraction later
    station.years = station_df["Year"].fillna(MDI).to_numpy()
    station.months = station_df["Month"].fillna(MDI).to_numpy()
    station.days = station_df["Day"].fillna(MDI).to_numpy()
    station.hours = station_df["Hour"].fillna(MDI).to_numpy()

    return station, station_df # read_station

#************************************************************************
def write_psv(outfile, df, separator, compression="infer"):
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    :param str separator: separating character (e.g. ",", "|")
    '''
    df.to_csv(outfile, index=False, sep=separator, compression=compression)

    return # write_psv

#************************************************************************
def write(outfile, df):
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    """

    # for .psv
    write_psv(outfile, df, "|")

    return # write

#************************************************************************
def flag_write(outfilename, df, diagnostics=False):
    """
    Write out flag summary files to enable quicker plotting

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    """
    with open(outfilename, "w") as outfile:

        for var in setup.obs_var_list:

            flags = df["{}_QC_flag".format(var)].fillna("")

            for test in QC_TESTS.keys():
                locs = flags[flags.str.contains(test)]

                outfile.write("{} : {} : {}\n".format(var, test, locs.shape[0]/flags.shape[0]))
                outfile.write("{} : {} : {}\n".format(var, "{}_counts".format(test), locs.shape[0]))


            # for total, get number of nonclean obs
            flagged, = np.where(flags != "")
            outfile.write("{} : {} : {}\n".format(var, "All", flagged.shape[0]/flags.shape[0]))
            outfile.write("{} : {} : {}\n".format(var, "{}_counts".format("All"), flagged.shape[0]))

            if diagnostics:
                print("{} - {}".format(var, flagged.shape[0]))

    return # flag_write
