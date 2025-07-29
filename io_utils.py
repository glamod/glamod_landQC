#!/usr/bin/env python
'''
io_utils - contains scripts for read/write of main files
'''
import os
import errno
import pandas as pd
import numpy as np
import setup
import datetime as dt
import csv
import subprocess
import shlex
import warnings
import logging
logger = logging.getLogger(__name__)

from qc_utils import Station, populate_station, MDI, QC_TESTS

#************************************************************************
def count_skip_rows(infile: str) -> list:
    """
    Read through the file, counting matches for expected header,
    but in unexpected lines (!=0).  Return these line numbers as list (zero-indexed)

    :param infile str: file to process

    :returns: list of line numbers
    """
    skip_rows = []
    with open(infile, "r") as data_file:
        reader = csv.reader(data_file)
        for r, row in enumerate(reader):
            if r == 0:
                continue
            if "STATION|Station_name" in row[0]:
                logger.warning(f"Extra header row at line {r}")
                skip_rows += [r]

    return skip_rows


#************************************************************************
def read_psv(infile: str, separator: str) -> pd.DataFrame:
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    https://stackoverflow.com/questions/64302419/what-are-all-of-the-exceptions-that-pandas-read-csv-throw

    :param str infile: location and name of infile (without extension)
    :param str separator: separating character (e.g. ",", "|")

    :returns: df - DataFrame
    '''
    warnings.filterwarnings("error", category=pd.errors.DtypeWarning)
    try:
        df = pd.read_csv(infile, sep=separator, compression="infer",
                         dtype=setup.DTYPE_DICT, na_values="Null", quoting=3, index_col=False)
    except FileNotFoundError as e:
        logger.warning(f"psv file not found: {str(e)}")
        print(str(e))
        raise FileNotFoundError(str(e))
    except ValueError as e:
        logger.warning(f"Error in psv rows: {str(e)}")
        print(str(e))
        # Presuming that there is an extra header line somewhere in the file

        # Find location of the extra header line
        skip_rows = count_skip_rows(infile)

        # Now re-read the file
        df = pd.read_csv(infile, sep=separator, compression="infer",
                         dtype=setup.DTYPE_DICT, na_values="Null", quoting=3,
                         index_col=False, skiprows=skip_rows)

    except pd.errors.ParserError as e:
        logger.warning(f"Parser Error: {str(e)}")
        print(str(e))
        raise pd.errors.ParserError(str(e))
    except EOFError as e:
        logger.warning(f"End of File Error (gzip): {str(e)}")
        print(str(e))
        raise EOFError(str(e))
    except pd.errors.DtypeWarning as e:
        logger.warning(f"Dtype error - likely header row missing: {str(e)}")
        print(str(e))
        raise RuntimeError

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
        df = read_psv(infile, "|")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), infile)

    # check there was a header row
    if df.columns[0] != "STATION":
        raise RuntimeError(f"Missing header row in {infile}")

    return df # read


#************************************************************************
def calculate_datetimes(station_df: pd.DataFrame) -> pd.Series:
    """
    Convert the separate Y-M-D H-M values into datetime objects

    :param pd.DataFrame station_df: dataframe for the station record

    :returns: pd.Series of datetime64 values
    """

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
                    print(f"Bad date: {yy}-{month[y]}-{day[y]}\n")
                    logger.warning(f"Bad date: {yy}-{month[y]}-{day[y]}\n")
                    raise ValueError(f"Bad date - {yy}-{month[y]}-{day[y]}")

    return datetimes


#************************************************************************
def convert_wind_flags(station_df: pd.DataFrame) -> None:

    # explicitly remove any missing data indicators - wind direction only
    for wind_flag in ["C-Calm", "V-Variable"]:
        combined_mask = (station_df["wind_direction_Measurement_Code"] == wind_flag) &\
                        (station_df["wind_direction"] == 999)
        station_df.loc[combined_mask, "wind_direction"] = np.nan


#************************************************************************
def read_station(stationfile: str, station: Station,
                 read_flags: bool = False) -> tuple[Station, pd.DataFrame]:
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
    except FileNotFoundError:
        logger.warning(f"Missing station file {stationfile}")
        raise FileNotFoundError
    except RuntimeError:
        logger.warning(f"Missing header row in {stationfile}")
        raise RuntimeError

    # calculate datetime series
    datetimes = calculate_datetimes(station_df)

    # convert any remaining wind flags
    convert_wind_flags(station_df)

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
def integrity_check(infile: str) -> bool:
    """Test integrity of a Gzip file

    Parameters
    ----------
    infile : str
        File path to test integrity of gzip file

    Returns
    -------
    bool
        Boolean indicator whether infile is valid gzip.
    """
    # using shlex to ensure safe usage of subprocess
    try:
        cmd = shlex.split(f"gzip -t {shlex.quote(infile)}")
        proc = subprocess.run(cmd,
                              capture_output=True, text=True, check=True, shell=False)
        logging.info(f"{proc.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        # Error raised if non-zero exit code
        logging.info(f"Validation failed on {infile}")
        logging.info(f"  {e.cmd}")
        logging.info(f"  {e.stderr}")
        return False



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

        # Latitude & Longitude = {:7.4f}
        # Monthy, Day, Hour, & Minute = {:0.2d}

    # for .psv
    write_psv(outfile, df, "|")

    if not integrity_check(outfile):
        logging.warning(f"Invalid Gzip file {outfile}")

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

            flags = df[f"{var}_QC_flag"].fillna("")

            # Pull out the actual observations
            this_var_data = df[var].fillna(MDI).to_numpy().astype(float)
            this_var_data = np.ma.masked_where(this_var_data == MDI, this_var_data)

            # write out for all tests, regardless if set for this variable or not
            for test in QC_TESTS.keys():
                locs = flags[flags.str.contains(test)]

                # For percentage, compare against all obs, not obs for that var
                if np.ma.count(this_var_data) == 0:
                    outfile.write(f"{var} : {test} : 0\n")
                else:
                    outfile.write(f"{var} : {test} : {locs.shape[0]/np.ma.count(this_var_data)}\n")
                outfile.write(f"{var} : {test}_counts : {locs.shape[0]}\n")


            # for total, get number of set flags (excluding fixable wind logical)
            flagged, = np.where(np.logical_and(flags != "", flags != "1"))
            if np.ma.count(this_var_data) == 0:
                outfile.write(f"{var} : All : 0\n")
            else:
                outfile.write(f"{var} : All : {flagged.shape[0]/np.ma.count(this_var_data)}\n")

            outfile.write(f"{var} : All_counts : {flagged.shape[0]}\n")

            logging.info(f"{var} - {flagged.shape[0]}")
            if diagnostics:
                print(f"{var} - {flagged.shape[0]}")
                print(f"{var} - {100*flagged.shape[0]/np.ma.count(this_var_data):.1f}%")

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
    outfilename = os.path.join(setup.SUBDAILY_ERROR_DIR, f"{station.id:11s}.err")

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
