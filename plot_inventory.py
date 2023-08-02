#!/usr/local/bin python


import os
import datetime as dt
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
# use the Agg environment to generate an image rather than outputting to screen
mpl.use('Agg')
import matplotlib.pyplot as plt


import setup
import qc_utils as utils

# what is available
START_YEAR = 1800
END_YEAR = dt.date.today().year

DAYS_IN_AVERAGE_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

TODAY = dt.datetime.strftime(dt.datetime.now(), "%Y%m%d")


#*********************************************
def read_stations():
    """
    Process the GHCNH history file
    
    Select stations which have defined lat, lon and elevation;
    at least N years of data (using start/end dates).  Also do additional
    processing for Canadian (71*) and German (09* and 10*) stations.

    :returns: list of station objects
    """

    fieldwidths = (7, 6, 30, 5, 3, 5, 8, 9, 8, 9, 9)

    all_stations = []

    try:
        # process the station list
        station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5), 
                               header=None, names=("id", "latitude", "longitude", "elevation", "state", "name", "wmo"))

        station_IDs = station_list.id
        for st, station_id in enumerate(station_IDs):
            # print("{} {:11s} ({}/{})".format(dt.datetime.now(), station_id, st+1, station_IDs.shape[0]))

            station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st],
                                    station_list.elevation[st])
 
            if station.lat != "" and station.lon != "" and station.elev != "" and float(station.elev) != -999.9:

                # test if elevation is known (above Dead-Sea shore in Jordan at -423m)
                if float(station.elev) > -430.0:

                    station.name = station_list.name[st]
                    station.wmo = station_list.wmo[st]
                    station.mergers = []
                    all_stations += [station]


    except OSError:
        print("{:s} does not exist.  Check download".format(GHCNH_LISTING))
        raise OSError
    
    print("{} stations in full GHCNH".format(len(station_IDs)))

    print("{} stations with defined metadata".format(len(all_stations)))
  
    return np.array(all_stations) # read_stations


#*********************************************
def extract_inventory(station, inventory, data_start, data_end, do_mergers=True):
    """
    Extract the information from the ISD inventory file

    :param station station: station object to extract data for
    :param array inventory: the full ISD inventory to extract information from
    :param int data_start: first year of data
    :param int data_end: last year of data
    :param bool do_mergers: include the information from merged stations when doing selection cuts
    :returns: array of Nyears x 12 months containing number of observation records in each month
    """

    monthly_obs = np.zeros([END_YEAR - START_YEAR + 1, 12])

    # are WMO IDs sufficient?
    locs, = np.where(inventory[:, 0] == station.id)

    if len(locs) == 0:
        return []
    else:
        this_station = inventory[locs, :]

        locs, = np.where(this_station[:, 2].astype(int) != 0)
        station.start = this_station[locs[0], 1].astype(int)
        station.end = this_station[locs[-1], 1].astype(int)
        
        for year in this_station:
            monthly_obs[int(year[1]) - START_YEAR, :] = year[3:].astype(int)

        if do_mergers:
            if station.mergers != []:

                for mstation in station.mergers:

                    merger_station = inventory[inventory[:, 0] == mstation, :]

                    for year in merger_station:
                        # only replace if have more observations
                        # can't tell how the merge will work with interleaving obs, so just take the max
                        locs = np.where(year[3:].astype(int) > monthly_obs[int(year[1]) - START_YEAR, :])
                        monthly_obs[int(year[1]) - START_YEAR, locs] = year[3:][locs].astype(int)

        # if restricted data period
        if data_start != START_YEAR:
            monthly_obs = monthly_obs[(data_start - START_YEAR):, :]

        if data_end != END_YEAR:
            monthly_obs = monthly_obs[:(data_end - END_YEAR), :]

        return monthly_obs # extract_inventory


#*********************************************
def process_inventory(candidate_stations, data_start, data_end):

    """
    Process the ISD inventory file

    :param list candidate_stations: list of station objects to match up
    :param int data_start: first year of data
    :param int data_end: last year of data
    """

    print("reading GHCNH inventory")

    # read in the inventory
    try:
        inventory = np.genfromtxt(os.path.join(setup.SUBDAILY_METADATA_DIR, setup.INVENTORY),
                                  skip_header=8, dtype=str)
    except OSError:
        pass

    all_counts = np.zeros((len(candidate_stations), 12*(data_end-data_start+1)))
    name_labels = []
    last_station = "-"
    # spin through each station
    for s, station in enumerate(candidate_stations):
        # print("{}/{}".format(s, len(candidate_stations)))

        if station.id[0] != last_station:
            name_labels += [[station.id[0], f"{s}"]]
            last_station = station.id[0]
            print(last_station)

        # extract the observations in each month for this station
        monthly_obs = extract_inventory(station, inventory, data_start, data_end, do_mergers=True)

        if len(monthly_obs) != 0:
            all_counts[s, :] = monthly_obs.reshape(-1)

#        if s > 1000: break

    # hide the zero-count months
    all_counts = np.ma.masked_where(all_counts <= 0, all_counts)

    name_labels = np.array(name_labels)

    # now plot this
    station_counter = np.arange(len(candidate_stations))
    years = np.arange(data_start, data_end+1, 1/12)

    print("plotting GHCNH inventory")
    plt.figure(figsize=(8, 25))
    plt.clf()
    ax1 = plt.axes([0.1, 0.02, 0.84, 0.95])
    plt.pcolormesh(years, station_counter, all_counts, cmap=plt.cm.viridis,
                   norm=mpl.colors.Normalize(vmin=0, vmax=1000))
    cb = plt.colorbar(orientation="horizontal", label="Monthly obs counts",
                      extend="max", ticks=np.arange(0, 1100, 100),
                      pad=0.02, aspect=30, fraction=0.02)
    
    plt.ylabel("Station sequence number")

    ax2 = ax1.twinx()
    ax2.set_ylim([None, len(station_counter)])
    ax2.set_yticks(name_labels[:, 1].astype(int), name_labels[:, 0])
    ax2.set_ylabel("Station start letter")

    plt.savefig(os.path.join(setup.SUBDAILY_IMAGE_DIR, "station_plot_{}.png".format(setup.DATESTAMP[:-1])), dpi=300)
                        
    return # process_inventory


if __name__ == "__main__":

    # parse text file into candidate list, with lat, lon, elev and time span limits applied
    all_stations = read_stations()
   
    data_start = 1800
    data_end = dt.datetime.now().year
    
    process_inventory(all_stations, data_start, data_end)
