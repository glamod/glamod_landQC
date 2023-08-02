#*********************************************
#  Make inventory listing of latest C3S run
#    of counts in each month of the station
#    record.
#
#*********************************************

import os
import sys
import configparser
import pandas as pd
import numpy as np
import datetime as dt
import calendar

import setup
import qc_utils as utils

MDI = -1.e30
month_names = [c.upper() for c in calendar.month_abbr[:]]
month_names[0] = "ANN"

TODAY = dt.datetime.now()
TODAY_MONTH = dt.datetime.strftime(TODAY, "%B").upper()

#*********************************************
def get_station_list(restart_id="", end_id=""):
    """
    Read in station list file(s) and return dataframe

    :param str restart_id: which station to start on
    :param str end_id: which station to end on

    :returns: dataframe of station list
    """

    # process the station list
    station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5), 
                               header=None, names=("id", "latitude", "longitude", "elevation", "state", "name", "wmo"))

    station_IDs = station_list.id

    # work from the end to save messing up the start indexing
    if end_id != "":
        endindex, = np.where(station_IDs == end_id)
        station_list = station_list.iloc[: endindex[0]+1]

    # and do the front
    if restart_id != "":
        startindex, = np.where(station_IDs == restart_id)
        station_list = station_list.iloc[startindex[0]:]

    return station_list.reset_index() # get_station_list

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

    # store extra information to enable easy extraction later
    station.years = station_df["Year"].fillna(MDI).to_numpy()
    station.months = station_df["Month"].fillna(MDI).to_numpy()
    station.days = station_df["Day"].fillna(MDI).to_numpy()
    station.hours = station_df["Hour"].fillna(MDI).to_numpy()

    return station, station_df # read_station

# ------------------------------------------------------------------------
# process the station list
def main(restart_id="", end_id="", clobber=False):
    """
    Main script.  Makes inventory and other metadata files

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool clobber: overwrite output file if exists
    """
    
    
    # write the headers
    with open(os.path.join(setup.SUBDAILY_METADATA_DIR, setup.INVENTORY), "w") as outfile:

        outfile.write("              *** GLOBAL HISTORICAL CLIMATE NETWORK HOURLY DATA INVENTORY ***\n")
        outfile.write("\n")
        outfile.write("THIS INVENTORY SHOWS THE NUMBER OF WEATHER OBSERVATIONS BY STATION-YEAR-MONTH FOR BEGINNING OF RECORD\n")
        outfile.write(f"THROUGH {TODAY_MONTH} {TODAY.year}.  THE DATABASE CONTINUES TO BE UPDATED AND ENHANCED, AND THIS INVENTORY WILL BE \n")
        outfile.write("UPDATED ON A REGULAR BASIS. QUALITY CONTROL FLAGS HAVE NOT BEEN INCLUDED IN THESE COUNTS, ALL OBSERVATIONS.\n")
        outfile.write("\n")
        month_string = " ".join([f"{c:<6s}" for c in month_names])
        outfile.write("{:11s} {:4s} {:84s}\n".format("STATION", "YEAR", month_string))
        outfile.write("\n")
 


        # read in the station list
        station_list = get_station_list(restart_id=restart_id, end_id=end_id)

        station_IDs = station_list.id

        # now spin through each ID in the curtailed list
        for st, station_id in enumerate(station_IDs):
            print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

            station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st], station_list.elevation[st])

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

            if len(station.years) == 0:
                print(f"{station} has no data")
                continue

            for year in range(station.years[0], TODAY.year):

                this_year_count = np.zeros(13)
                for month in range(1, 13):

                    locs, = np.where(np.logical_and(station.years==year, station.months==month))

                    this_year_count[month] = len(locs)

                # also store annual count
                this_year_count[0] = this_year_count.sum()

                year_string = " ".join([f"{c:<6.0f}" for c in this_year_count])

                outfile.write(f"{station_id:11s} {year:4.0f} {year_string:84s}\n")

            del(station_df)
            del(station)

    return
                

#************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart_id', dest='restart_id', action='store', default="",
                        help='Restart ID for truncated run, default=""')
    parser.add_argument('--end_id', dest='end_id', action='store', default="",
                        help='End ID for truncated run, default=""')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False,
                        help='Overwrite output files if they exists.')

    args = parser.parse_args()

    main(restart_id=args.restart_id,
         end_id=args.end_id,
         clobber=args.clobber,
    )

#************************************************************************
