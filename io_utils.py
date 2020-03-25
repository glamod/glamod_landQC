#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import pandas as pd
import setup

from qc_utils import populate_station, MDI

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

    :param str outfile: location and name of outfile (without extension)
    :param DataFrame df: data frame to write
    :param str separator: separating character (e.g. ",", "|")
    '''
    df.to_csv(outfile, index=False, sep=separator, compression=compression)

    return # write_psv

#************************************************************************
def write(outfile, df):
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile (without extension)
    :param DataFrame df: data frame to write
    """

    # for .psv
    write_psv(outfile, df, "|")

    return # write

