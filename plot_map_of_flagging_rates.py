#************************************************************************
import cartopy.crs as ccrs
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# internal utils
import qc_utils as utils
import io_utils as io
import qc_tests
import setup
#************************************************************************

QC_TESTS = {"o" : "Odd Cluster", "F" : "Frequent Value", "D" : "Distribution - Monthly", \
            "d" : "Distribution - all", "W" : "World Records", "K" : "Streaks", \
            "C" : "Climatological", "T" : "Timestamp", "S" : "Spike", "h" : "Humidity", \
            "V" : "Variance", "p" : "Pressure", "w" : "Winds", "L" : "Logic"}

TESTS_FOR_VARS = {"temperature": ["o", "F", "D", "d", "W", "K", "C", "T", "S", "V", "L"],\
                      "dew_point_temperature": ["o", "F", "D", "d", "W", "K", "C", "T", "S", "h", "V", "L"],\
                      "sea_level_pressure" : ["o", "F", "D", "d", "W", "K", "T", "S", "V", "p", "L"],\
                      "station_level_pressure" : ["o", "F", "D", "d", "K", "T", "S", "V", "p", "L"],\
                      "wind_speed" : ["o", "W", "K", "T", "S", "V", "w", "L"],
                  "wind_direction" : ["K", "w", "L"]}

# Temporary stuff
MFF_LOC = setup.SUBDAILY_IN_DIR
QFF_LOC = setup.SUBDAILY_OUT_DIR
CONF_LOC = setup.SUBDAILY_CONFIG_DIR
IMAGE_LOCS = setup.SUBDAILY_IMAGE_DIR

start_time_string = dt.datetime.strftime(dt.datetime.now(), "%Y%m%d")

#************************************************************************
def main(restart_id="", end_id="", diagnostics=False):
    """
    Main plot function.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    """

    obs_var_list = setup.obs_var_list

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    station_IDs = station_list.iloc[:, 0]

    all_stations = {}

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):
        print("{} {}".format(dt.datetime.now(), station_id))

#        if station_id not in ["AFA00409803", "AFA00409793"]: continue

        station = utils.Station(station_id, station_list.iloc[st][1], station_list.iloc[st][2], station_list.iloc[st][3])
        if diagnostics:
            print(station)

        #*************************
        # read QFF
        try:
            station_df = io.read(os.path.join(QFF_LOC, station_id), extension="qff")
        except IOError:
            print("Missing station {}".format(station_id))
            continue

        for var in obs_var_list:

            setattr(station, var, utils.Meteorological_Variable("{}".format(var), utils.MDI, "", ""))
            obs_var = getattr(station, var)

            flags = station_df["{}_QC_flag".format(var)].fillna("")

            for test in QC_TESTS.keys():
                locs = flags[flags.str.match(test)]

                setattr(obs_var, test, locs.shape[0]/flags.shape[0])

        all_stations[station_id] = station

    # now spin through each var/test combo and make a plot
    for var in obs_var_list:
        for test in TESTS_FOR_VARS[var]:

            lats, lons, flag_fraction = np.zeros(station_IDs.shape[0]), np.zeros(station_IDs.shape[0]), np.zeros(station_IDs.shape[0])
            for st, (ID, station) in enumerate(all_stations.items()):
                lats[st] = station.lat
                lons[st] = station.lon
                obs_var = getattr(station, var)
                flag_fraction[st] = getattr(obs_var, test) * 100               

            # do the plot
            plt.figure(figsize=(8, 5))
            plt.clf()
            ax = plt.axes([0, 0.03, 1, 1], projection=ccrs.Robinson())
            ax.set_global()
            ax.coastlines('50m')
            try:
                ax.gridlines(draw_labels = True)
            except TypeError:
                ax.gridlines()

            # colors are the exact same RBG codes as in IDL
            colors = [(150, 150, 150), (41, 10, 216), (63, 160, 255), (170, 247, 255), \
                      (255, 224, 153), (247, 109, 94), (165, 0, 33), (0, 0, 0)]
            limits = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 100.]

            for u, upper in enumerate(limits):

                # sort the labels
                if u == 0:
                    locs, = np.where(flag_fraction == 0)
                    label = "{}%: {}".format(upper, len(locs))
                else:
                    locs, = np.where(np.logical_and(flag_fraction <= upper, \
                                                    flag_fraction > limits[u-1]))
                    label = ">{} to {}%: {}".format(limits[u-1], upper, len(locs))
                    if upper == limits[-1]:
                        label = ">{}%: {}".format(limits[u-1], len(locs))

                # and plot
                if len(locs) > 0:
                    ax.scatter(lons[locs], lats[locs], transform = ccrs.Geodetic(), s = 15, c = tuple([float(c)/255 for c in colors[u]]), \
                               edgecolors="none", label = label)

                else:
                    ax.scatter([0], [-90], transform = ccrs.Geodetic(), s = 15, c = tuple([float(c)/255 for c in colors[u]]), \
                               edgecolors="none", label = label)

            plt.title("{} - {}".format(" ".join([v.capitalize() for v in var.split("_")]), QC_TESTS[test]))

            watermarkstring="/".join(os.getcwd().split('/')[4:])+'/'+\
                os.path.basename( __file__ )+"   "+dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
            plt.figtext(0.01,0.01,watermarkstring,size=5)

            leg=plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False, title='', prop={'size':9}, \
                           labelspacing=0.15, columnspacing=0.5, numpoints=1)

            plt.savefig(os.path.join(IMAGE_LOCS, "All_fails_{}-{}_{}.png".format(var, test, start_time_string)))
            plt.close()


    return # main

#************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart_id', dest='restart_id', action='store', default="",
                        help='Restart ID for truncated run, default=""')
    parser.add_argument('--end_id', dest='end_id', action='store', default="",
                        help='End ID for truncated run, default=""')
    parser.add_argument('--diagnostics', dest='diagnostics', action='store_true', default=False,
                        help='Run diagnostics (will not write out file)')

    args = parser.parse_args()
    main(restart_id=args.restart_id, end_id=args.end_id, diagnostics=args.diagnostics)

#*******************************************************
