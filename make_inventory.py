#*********************************************
#  Make inventory listing of latest C3S run
#    of counts in each month of the station
#    record.
#
#*********************************************

import os
import numpy as np
import datetime as dt
import calendar

import setup
import qc_utils as utils
import io_utils as ioutils

MDI = -1.e30
month_names = [c.upper() for c in calendar.month_abbr[:]]
month_names[0] = "ANN"

TODAY = dt.datetime.now()
TODAY_MONTH = dt.datetime.strftime(TODAY, "%B").upper()


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
        station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

        station_IDs = station_list.id

        # now spin through each ID in the curtailed list
        for st, station_id in enumerate(station_IDs):
            print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

            station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st], station_list.elevation[st])

            try:
                station, station_df = ioutils.read_station(os.path.join(setup.SUBDAILY_OUT_DIR, "{:11s}.{}{}".format(station_id, "qff", setup.OUT_COMPRESSION)), station)
            except OSError as e:
                # file missing, move on to next in sequence
                print(f"{station}, File Missing, {str(e)}")
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
