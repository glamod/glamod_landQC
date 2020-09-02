#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import pandas as pd
import numpy as np
import sys
import setup
import datetime as dt
        
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

    try:
        df = pd.read_csv(infile, sep=separator, compression=compression, dtype=setup.DTYPE_DICT, na_values="Null")
    except ValueError as e:
        print(str(e))
        raise ValueError(str(e))

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
        try:
            df = read_psv(infile, "|")
        except ValueError as e:
            raise ValueError(str(e))
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
        print("Missing station file {}".format(stationfile))
        raise OSError
    except ValueError as e:
        print("Issue in station file {}".format(stationfile))
        raise ValueError(str(e))


    # convert to datetimes
    try:
        datetimes = pd.to_datetime(station_df[["Year", "Month", "Day", "Hour", "Minute"]])
    except ValueError as e:
        if str(e) == "cannot assemble the datetimes: day is out of range for month":
            year = station_df["Year"]
            month = station_df["Month"]
            day = station_df["Day"]
            for y, yy in enumerate(year):
                try:
                    dummy = dt.datetime(yy, month[y], day[y])
                except ValueError:
                    print(yy, month[y], day[y])
                    print("Bad Date")
                    raise ValueError("Bad date - {}-{}-{}".format(yy, month[y], day[y]))

            

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
def write(outfile, df, formatters={}):
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    :param formatters dict: dictionary of formatters
    """
    # need to adjust formatting for certain columns before writing

    for column, fmt in formatters.items():
        print(key, value)
        df[column] = pd.Series([fmt.format(val) for val in df[column]], index = df.index)

#    df['Latitude'] = pd.Series(["{:7.4f}".format(val) for val in df['Latitude']], index = df.index)
#    df['Longitude'] = pd.Series(["{:7.4f}".format(val) for val in df['Longitude']], index = df.index)
#    df['Month'] = pd.Series(["{:02d}".format(val) for val in df['Month']], index = df.index)
#    df['Day'] = pd.Series(["{:02d}".format(val) for val in df['Day']], index = df.index)
#    df['Hour'] = pd.Series(["{:02d}".format(val) for val in df['Hour']], index = df.index)
#    df['Minute'] = pd.Series(["{:02d}".format(val) for val in df['Minute']], index = df.index)

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

#************************************************************************
def write_error(station, message, error="", diagnostics=False):
    """
    Write out quick failure message for station

    :param Station station: met. station 
    :param str message: message to store
    :param str error: error output from stacktrace
    :param bool diagnostics: turn on diagnostic output
    """
    outfilename = os.path.join(setup.SUBDAILY_ERROR_DIR, "{:11s}.err".format(station.id))

    with open(outfilename, "w") as outfile:
        outfile.write(dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M") + "\n")
        outfile.write(message + "\n")
        if error != "":
            outfile.write(error + "\n")
        
    return # write_error
