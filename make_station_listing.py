#*********************************************
#  Make new station list with start and end
#    dates for each station added
#
#*********************************************
import os
import numpy as np
import datetime as dt

import setup
import utils
import io_utils as ioutils

# ------------------------------------------------------------------------
# process the station list
def main() -> None:
    """
    Main script.  Makes inventory and other metadata files

    """

    # read in the station list
    station_list = utils.get_station_list()

    # insert the new columns
    station_list.insert(len(station_list.columns), "begins", ["" for i in range(station_list.shape[0])])
    station_list.insert(len(station_list.columns), "ends", ["" for i in range(station_list.shape[0])])

    station_IDs = station_list.id

    begins = np.array(["99999999" for i in range(station_list.shape[0])])
    ends = np.array(["99999999" for i in range(station_list.shape[0])])

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print(f"{dt.datetime.now()} {station_id:11s} ({st+1}/{station_IDs.shape[0]})")

        station = utils.Station(station_id, station_list.latitude[st],
                                station_list.longitude[st], station_list.elevation[st])

        try:
            station, station_df = ioutils.read_station(os.path.join(setup.SUBDAILY_OUT_DIR,
                                                                    f"{station_id:11s}{setup.OUT_SUFFIX}{setup.OUT_COMPRESSION}"),
                                                       station)
        except OSError as e:
            # file missing, move on to next in sequence
            print(f"{station}, File Missing, {str(e)}")
            continue
        except ValueError as e:
            # some issue in the raw file
            print(f"{station}, Error in input file, {str(e)}")
            continue
        except EOFError as e:
            # some issue in the gzip archive
            print(f"{station}, Error in gzip file, {str(e)}")
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
                           index=False, header=False, index_names=False,
                           formatters={"wmo": "{:5s}".format,
                                       "name": "{:<40s}".format}, na_rep="")

    return

#************************************************************************
if __name__ == "__main__":

    main()
