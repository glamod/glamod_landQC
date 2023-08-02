#*********************************************
#  Make new station list with start and end
#    dates for each station added
#
#*********************************************
import os
import numpy as np
import pandas as pd
import datetime as dt

import setup
import qc_utils as utils


#*********************************************
def get_station_list():
    """
    Read in station list file(s) and return dataframe

    :returns: dataframe of station list
    """

    # process the station list
    station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5), 
                               header=None, names=("id", "latitude", "longitude", "elevation", "state", "name", "wmo"))

    return station_list

#************************************************************************
def read_psv(infile, separator):
    '''
    http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    :param str infile: location and name of infile (without extension)
    :param str separator: separating character (e.g. ",", "|")

    :returns: df - DataFrame
    '''

    try:
        df = pd.read_csv(infile, sep=separator, compression="infer", dtype=setup.DTYPE_DICT, na_values="Null")
    except ValueError as e:
        print(str(e))
        raise ValueError(str(e))

    return df #  read_psv

#************************************************************************
def read(infile):
    """
    Wrapper for read functions to allow remainder to be file format agnostic.

    :param str infile: location and name of infile (without extension)

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
    # read QFF
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
    utils.populate_station(station, station_df, setup.obs_var_list, read_flags=read_flags)
    station.times = datetimes

    return station, station_df # read_station


# ------------------------------------------------------------------------
# process the station list
def main():
    """
    Main script.  Makes inventory and other metadata files

    """
    
    # read in the station list
    station_list = get_station_list()

    # insert the new columns
    station_list.insert(len(station_list.columns), "begin", ["" for i in range(station_list.shape[0])])
    station_list.insert(len(station_list.columns), "end", ["" for i in range(station_list.shape[0])])

    station_IDs = station_list.id

    begins = np.array(["99999999" for i in range(station_list.shape[0])])
    ends = np.array(["99999999" for i in range(station_list.shape[0])])  
    
    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

        station = utils.Station(station_id, station_list.latitude[st],
                                station_list.longitude[st], station_list.elevation[st])

        try:
            station, station_df = read_station(os.path.join(setup.SUBDAILY_OUT_DIR, "{:11s}.{}{}".format(station_id, "qff", setup.IN_COMPRESSION)), station)
        except OSError as e:
            # file missing, move on to next in sequence
            print(f"{station}, File Missing")
            continue
        except ValueError as e:
            # some issue in the raw file
            print(f"{station}, Error in input file, {str(e)}")
            continue

        if len(station.times) == 0:
            begins[st] = "99999999"
            ends[st] = "99999999"
        else:
            begins[st] = dt.datetime.strftime(station.times.iloc[0], "%Y%m%d")
            ends[st] = dt.datetime.strftime(station.times.iloc[-1], "%Y%m%d")

    station_list["begins"] = begins
    station_list["ends"] = ends

    station_list.to_string(os.path.join(setup.SUBDAILY_METADATA_DIR, setup.STATION_FULL_LIST),
                           index=False, header=False,
                           formatters={"wmo": "{:05.0f}".format,
                                       "name": "{:<39s}".format}, na_rep="")
    
    return
    
#************************************************************************
if __name__ == "__main__":

    main()
