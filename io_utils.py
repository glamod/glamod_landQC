#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import pandas as pd
import numpy as np
import setup
import datetime as dt
import logging
logger = logging.getLogger(__name__)
        
from qc_utils import Station, populate_station, MDI, QC_TESTS

#************************************************************************
def read_psv(infile: str, separator: str) -> pd.DataFrame:
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str infile: location and name of infile (without extension)
    :param str separator: separating character (e.g. ",", "|")
 
    :returns: df - DataFrame
    '''

    try:
        df = pd.read_csv(infile, sep=separator, compression="infer", dtype=setup.DTYPE_DICT, na_values="Null", quoting=3)
    except ValueError as e:
        logger.warning(f"Error reading psv: {str(e)}")
        print(str(e))
        raise ValueError(str(e))

    # Number of columns at August 2023, or after adding flag columns
    assert len(df.columns) in [238, 238+len(setup.obs_var_list)]

    return df #  read_psv

#************************************************************************
def read(infile:str) -> pd.DataFrame:
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
def read_station(stationfile: str, station: Station, read_flags: bool = False) -> tuple[Station, pd.DataFrame]:
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
        logger.warning(f"Missing station file {stationfile}\n")
        raise OSError
    except ValueError as e:
        logger.warning(f"Missing station file {stationfile}\n")
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
                    # if Datatime doesn't throw an error here, then it's valid
                    _ = dt.datetime(yy, month[y], day[y])
                except ValueError:
                    logger.warning(f"Bad date: {yy}-{month[y]}-{day[y]}\n")
                    raise ValueError(f"Bad date - {yy}-{month[y]}-{day[y]}")

    # explicitly remove any missing data indicators - wind direction only
    for wind_flag in ["C-Calm", "V-Variable"]:
        combined_mask = (station_df["wind_direction_Measurement_Code"] == wind_flag) &\
                        (station_df["wind_direction"] == 999)
        station_df.loc[combined_mask, "wind_direction"] = np.nan

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
def write_psv(outfile: str, df: pd.DataFrame, separator: str) -> None:
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    :param str separator: separating character (e.g. ",", "|")
    '''
    df.to_csv(outfile, index=False, sep=separator, compression="infer")

    return # write_psv

#************************************************************************
def write(outfile: str, df: pd.DataFrame, formatters: dict = {}) -> None:
    """
    Wrapper for write functions to allow remainder to be file format agnostic.

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    :param formatters dict: dictionary of formatters
    """
    # need to adjust formatting for certain columns before writing

    for column, fmt in formatters.items():
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
def flag_write(outfilename: str, df: pd.DataFrame, diagnostics: bool = False) -> None:
    """
    Write out flag summary files to enable quicker plotting

    :param str outfile: location and name of outfile
    :param DataFrame df: data frame to write
    :param bool diagnostics: verbose output
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

            logging.info(f"{var} - {flagged.shape[0]}")
            if diagnostics:
                print(f"{var} - {flagged.shape[0]}")

    return # flag_write

#************************************************************************
def write_error(station: Station, message: str, error: str = "", diagnostics:bool = False) -> None:
    """
    Write out quick failure message for station

    :param Station station: met. station 
    :param str message: message to store
    :param str error: error output from stacktrace
    :param bool diagnostics: turn on diagnostic output
    """
    outfilename = os.path.join(setup.SUBDAILY_ERROR_DIR, "{:11s}.err".format(station.id))

    # in case this file already exists, then append
    if os.path.exists(outfilename):
        write_type = "a"
    else:
        write_type = "w"

    with open(outfilename, write_type) as outfile:
        outfile.write(dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M") + "\n")
        outfile.write(message + "\n")
        if error != "":
            outfile.write(error + "\n")

    return # write_error
