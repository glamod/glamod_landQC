import os
import datetime as dt
import argparse
import numpy as np
import pandas as pd
import matplotlib
# use the Agg environment to generate an image rather than outputting to screen
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# RJHD Utilities
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

    :returns: list of selected station objects
    """
    all_stations = []

    try:
        # process the station list
        station_list = pd.read_fwf(setup.STATION_LIST, widths=(11, 9, 10, 7, 3, 40, 5), 
                               header=None, names=("id", "latitude", "longitude", "elevation", "state", "name", "wmo"))

        station_IDs = station_list.id
        for st, station_id in enumerate(station_IDs):
            station = utils.Station(station_id, station_list.latitude[st], station_list.longitude[st],
                                    station_list.elevation[st])
 
            if station.lat != "" and station.lon != "" and station.elev != "" and float(station.elev) != -999.9:

                # test if elevation is known (above Dead-Sea shore in Jordan at -423m)
                if float(station.elev) > -430.0:

                    station.name = station_list.name[st]
                    station.wmo = station_list.wmo[st]
                    all_stations += [station]


    except OSError:
        print("{:s} does not exist.  Check download".format(setup.STATION_LIST))
        raise OSError
    
    print("{} stations in full GHCNH".format(len(station_IDs)))

    print("{} stations with defined metadata".format(len(all_stations)))
  
    return np.array(all_stations) # read_stations


#*********************************************
def extract_inventory(station, inventory):
    """
    Extract the information from the ISD inventory file

    :param station station: station object to extract data for
    :param array inventory: the full ISD inventory to extract information from

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

        return monthly_obs # extract_inventory


#*********************************************
def process_inventory(candidate_stations):

    """
    Process the ISD inventory file

    Use input list to select those which have at least 6 hourly reporting over min_years_present years.

    :param list candidate_stations: list of station objects to match up

    :returns: list of selected station objects
    """
    

    print("filtering GHCNH inventory")

    # read in the inventory
    try:
        inventory = np.genfromtxt(os.path.join(setup.SUBDAILY_METADATA_DIR, setup.INVENTORY),
                                  skip_header=8, dtype=str)
    except OSError:
        pass

    present_stations = []
    long_stations = []
    final_stations = []
    # spin through each station
    for s, station in enumerate(candidate_stations):
        # print("{}/{}".format(s, len(candidate_stations)))

        # extract the observations in each month for this station
        monthly_obs = extract_inventory(station, inventory)

        if len(monthly_obs) != 0:
            present_stations += [station]

            station.obs = monthly_obs

    inventory = [] #  clear memory    

    return present_stations # process_inventory

#*********************************************
def stations_per_year(candidate_stations):
    """
    Return list of which years each station is present in

    :param list candidate_station: list of station objects

    :returns: stations_active_in_years - list of lists - which years a station is active in.   

    """

    # set up blank lists
    stations_active_in_years = [[] for stn in candidate_stations]

    # spin through stations
    for s, station in enumerate(candidate_stations):

        # process each year at a time - using .obs attribute (monthly_obs from extract_inventory)
        for y, year in enumerate(station.obs):

            if np.sum(year) != 0:

                stations_active_in_years[s] += [y + START_YEAR]

    return stations_active_in_years # stations_per_year

#*********************************************
def plot_stations(station_list, outfile, title=""):
    """
    Plot the stations on a global map

    :param list station_list: list of station objects
    :param str outfile: name of output file
    :param str title: plot title
    :returns:
    """


    plt.figure(figsize=(8, 5))
    plt.clf()
    ax = plt.axes([0.05, 0.07, 0.9, 0.9], projection=ccrs.Robinson())
    ax.coastlines('50m')
    try:
        ax.gridlines(color="black", draw_labels=True)
    except TypeError:
        ax.gridlines(color="black")

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, color="lightblue")

    ax.set_extent([-180.1, 180.1, -90, 90], crs=ccrs.PlateCarree())

    lats, lons = [], []
    mlats, mlons = [], []

    for stn in station_list:

        lats += [stn.lat]
        lons += [stn.lon]

    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), s=1, c='midnightblue',
               edgecolor='midnightblue', label='GHCNH {} stations'.format(TODAY))

    ax.set_global()
 
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.17), frameon=False, prop={'size':13})
    if title != "":
        plt.suptitle("{} - {} stations".format(title, len(lats)))
    else:
        plt.suptitle("{} stations".format(len(lats)))

    watermarkstring = dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
    plt.figtext(0.01, 0.01, watermarkstring, size=5)

    plt.savefig(outfile)
    plt.close()

    return # plot_stations


#****************************************************
def plot_gridded_map(station_list, outfile, title=""):

    #*************************
    # plot a gridded map

    # gridcell size
    delta_lon, delta_lat = 1.5, 1.5
    gridmax = 20.

    # create array of cell boundaries and make grid
    rawlons = np.arange(-180., 180.+delta_lon, delta_lon)
    rawlats = np.arange(-90., 90.+delta_lat, delta_lat)
    gridlon, gridlat = np.meshgrid(rawlons, rawlats)
    # set up empty array for gridded data
    griddata = np.zeros(list(gridlon.shape))

    UsedStation = np.zeros(len(station_list))

    lats, lons = [], []
    for stn in station_list:
        lats += [stn.lat]
        lons += [stn.lon]

    nstations = len(station_list)

    StationNumbers = np.zeros(gridlon.shape)
    start = 0
    for tlons, longitude in enumerate(gridlon[0, 1:]):

        for tlats, latitude in enumerate(gridlat[1:, 0]):

            for st in range(start, nstations):
                 if UsedStation[st] != 1.:
                    # if station not already in a grid box
                    if lats[st] < latitude and \
                            lons[st] < longitude:
                        """counts from bottom LH corner upwards
                        so starts at -180, -90.
                        Grid box values are to the top right of the
                        coordinates, with the final set ignored
                        This counts the stations to the bottom left
                        so do an offset to "place" result in "correct"
                        grid box.  No stations are "<" -180, -90
                        """
                        StationNumbers[tlats+1, tlons+1] += 1
                        UsedStation[st] = 1.

    print("Total number of stations passing criteria ", np.sum(StationNumbers))
    print("Grid maximum set to "+str(gridmax))

    plt.figure(figsize=(8, 5))
    plt.clf()
    ax = plt.axes([0.05, 0.08, 0.9, 0.9], projection=ccrs.Robinson())
    ax.coastlines('50m')
    try:
        ax.gridlines(color="black", draw_labels=True)
    except TypeError:
        ax.gridlines(color="black")

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, color="lightblue")
    ax.set_extent([-180.1, 180.1, -90, 90], crs=ccrs.PlateCarree())
    # mask the grid for zeros.
    StationNumbers[StationNumbers > gridmax] = gridmax
    MaskedGriddata = np.ma.masked_where(StationNumbers == 0, StationNumbers)

    # plot the grid - set max for colourbar at 50
    cs = plt.pcolormesh(gridlon, gridlat, MaskedGriddata, cmap=plt.cm.viridis,
                        alpha=0.8, vmax=gridmax, vmin=1, transform=ccrs.PlateCarree())

    # colour bar
    cb = plt.colorbar(cs, orientation='horizontal', pad=0.04, fraction=0.04,
                      ticks=[1, 5, 10, 15, 20], shrink=0.8)
    cb.set_label('Number of Stations')

    plt.suptitle(title)

    watermarkstring = dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
    plt.figtext(0.01, 0.01, watermarkstring, size=5)
    #plt.show()

    plt.savefig(outfile)
    return # plot_gridded_map


#****************************************************
def plot_station_number_over_time(station_list, outfile):
    """
    Plot the number of stations in each year

    :param list station_list: list of station objects

    :param str outfile: name of output file

    :returns:
    """
    # flatten list
    stations_active_in_years = stations_per_year(station_list)
    station_years = np.array([item for sublist in stations_active_in_years for item in list(set(sublist))])

    plt.clf()
    station_numbers = []

    for y in range(START_YEAR, END_YEAR+1):
        station_numbers += [len(station_years[station_years == y])]

    plt.plot(np.arange(START_YEAR, END_YEAR+1), station_numbers, 'co', ls='-', label="raw")

    # prettify the plot
    plt.ylim([0, 20000])
    plt.ylabel("Stations with data")
    plt.xlim([START_YEAR, END_YEAR+2])
    plt.xticks(np.arange(START_YEAR, END_YEAR+2, 20))
    watermarkstring = dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
    plt.figtext(0.01, 0.01, watermarkstring, size=5)

    plt.savefig(outfile+".png")

    # and log-scale y-axis
    plt.clf()
    station_numbers = []

    for y in range(START_YEAR, END_YEAR+1):
        station_numbers += [len(station_years[station_years == y])]

    station_numbers = np.array(station_numbers)
    station_numbers = np.ma.masked_where(station_numbers == 0, station_numbers)
        
    plt.plot(np.arange(START_YEAR, END_YEAR+1), station_numbers, 'co', ls='-', label="raw")

    # prettify the plot
    plt.ylim([1, 20000])
    plt.gca().set_yscale("log")
    plt.ylabel("Stations with data")
    plt.xlim([START_YEAR, END_YEAR+2])
    plt.xticks(np.arange(START_YEAR, END_YEAR+2, 20))
    watermarkstring = dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
    plt.figtext(0.01, 0.01, watermarkstring, size=5)

    plt.savefig(outfile+"_log.png")

    return station_numbers # plot_station_number_over_time


#*********************************************
def main():

    # parse text file into candidate list, with lat, lon, elev and time span limits applied
    all_stations = read_stations()      
    plot_stations(all_stations,
                  os.path.join(setup.SUBDAILY_IMAGE_DIR, 'ghcnh_station_distribution_{}.png'.format(TODAY)),
                  title="GHCNh Stations")

    # use inventory to further refine station list
    candidate_stations = process_inventory(all_stations)

    # plot distribution of stations
    plot_station_number_over_time(candidate_stations,
                                  os.path.join(setup.SUBDAILY_IMAGE_DIR, 'ghcnh_station_number_{}'.format(TODAY))) 

    plot_gridded_map(candidate_stations,
                     os.path.join(setup.SUBDAILY_IMAGE_DIR, 'ghcnh_gridded_station_distribution_{}.png'.format(TODAY)),
                     title="GHCNh Stations")

    # plot station numbers in all years
    for year in range(START_YEAR, END_YEAR + 1):
        plot_list = []
        # spin through all stations
        for stn in candidate_stations:
            # if at least one observation in that year, select it.
            if np.sum(stn.obs[year - START_YEAR]) > 0:
                plot_list += [stn]

        plot_stations(plot_list,
                      os.path.join(setup.SUBDAILY_IMAGE_DIR, "ghcnh_station_number_in_{}_{}.png".format(year, TODAY)),
                      title=str(year)), 
        print(year, len(plot_list))

    return

#*******************
if __name__ == "__main__":

    main()

#------------------------------------------------------------
# END
#------------------------------------------------------------
