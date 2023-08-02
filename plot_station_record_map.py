#*********************************************
#  Plot map of station record lengths
#
#*********************************************
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl

# use the Agg environment to generate an image rather than outputting to screen
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import setup

#*********************************************
def get_station_list():
    """
    Read in station list file(s) and return dataframe

    :returns: dataframe of station list
    """

    # process the station list
    station_list = pd.read_fwf(os.path.join(setup.SUBDAILY_METADATA_DIR, setup.STATION_FULL_LIST),
                               widths=(11, 9, 10, 7, 3, 41, 5, 11, 9), 
                               header=None, names=("id", "latitude", "longitude", "elevation",
                                                   "state", "name", "wmo", "begins", "ends"))

    return station_list


# ------------------------------------------------------------------------
# process the station list
def main():
    """
    Main script.  Makes inventory and other metadata files

    """
    
    # read in the station list
    station_list = get_station_list()

    length = np.zeros(station_list.shape[0])
    lats = station_list.latitude
    lons = station_list.longitude

    for st, station in enumerate(station_list.id):
        if station_list.begins[st] != 99999999:
 
            start = dt.datetime.strptime(station_list.begins[st].astype(str), "%Y%m%d")
            end = dt.datetime.strptime(station_list.ends[st].astype(str), "%Y%m%d")
            
            length[st] = (end-start).days/365

   
    plt.figure(figsize=(8, 5))
    plt.clf()
    ax = plt.axes([0.05, 0.09, 0.9, 0.9], projection=ccrs.Robinson())
    ax.coastlines('50m')
    try:
        ax.gridlines(color="black", draw_labels=True)
    except TypeError:
        ax.gridlines(color="black")

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, color="lightblue")

    ax.set_global()
    
    scat = ax.scatter(lons, lats, transform=ccrs.PlateCarree(), s=1, c=length,
               cmap = plt.cm.viridis, norm=mpl.colors.Normalize(vmin=0, vmax=100))

    cb = plt.colorbar(scat, orientation="horizontal", label="Years of record",
                      extend="max", ticks=np.arange(0, 110, 10),
                      pad=0.06, aspect=30, fraction=0.05)


    plt.savefig(os.path.join(setup.SUBDAILY_IMAGE_DIR, "station_map_record_length.png"))

    return

#************************************************************************
if __name__ == "__main__":

    main()
