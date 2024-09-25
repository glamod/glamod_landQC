#************************************************************************
import cartopy.crs as ccrs
import os
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# internal utils
import qc_utils as utils
import setup
#************************************************************************


TESTS_FOR_VARS = {"temperature": ["All", "o", "U", "F", "D", "d", "W", "K", "C", "T", "S", "V", "L", "E", "N", "H"],\
                      "dew_point_temperature": ["All", "o", "F", "D", "d", "W", "K", "C", "T", "S", "h", "V", "L", "E", "N", "H", "n"],\
                      "sea_level_pressure" : ["All", "o", "F", "D", "d", "W", "K", "T", "S", "V", "p", "L", "E", "N", "H"],\
                      "station_level_pressure" : ["All", "o", "F", "D", "d", "K", "T", "S", "V", "p", "L", "E", "N", "H"],\
                      "wind_speed" : ["All", "o", "W", "K", "T", "S", "V", "w", "L", "E", "N", "H"],
                  "wind_direction" : ["All", "K", "w", "L", "E", "H"]}

UNITS = {"" : "%", "_counts" : "cts"}


# Temporary stuff
MFF_LOC = setup.SUBDAILY_MFF_DIR
QFF_LOC = setup.SUBDAILY_OUT_DIR
CONF_LOC = setup.SUBDAILY_CONFIG_DIR
IMAGE_LOCS = setup.SUBDAILY_IMAGE_DIR

start_time_string = dt.datetime.strftime(dt.datetime.now(), "%Y%m%d")


#************************************************************************
def flag_read(infilename: str) -> None:
    """
    Read flag summary file into dictionary of dicts
    """

    flags = {}

    all_flags = np.genfromtxt(infilename, delimiter=":", dtype=(str))

    # spin through and build up dict of dicts
    for var, test, value in all_flags:
        flags.setdefault(var.strip(), {})[test.strip()] = float(value)

    return flags # flag_read

#************************************************************************
def main(restart_id: str = "", end_id: str = "", diagnostics: bool = False) -> None:
    """
    Main plot function.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    """

    obs_var_list = setup.obs_var_list

    # process the station list
    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    station_IDs = station_list.id

    all_stations = {}

    # now spin through each ID in the curtailed list
    for st, station_id in enumerate(station_IDs):

#        if st > 10:
#            break
        
        print(f"{dt.datetime.now()} {station_id}")

        station = utils.Station(station_id, station_list.iloc[st].latitude, station_list.iloc[st].longitude, station_list.iloc[st].elevation)
        if diagnostics:
            print(station)

        try:
            flag_summary = flag_read(os.path.join(setup.SUBDAILY_FLAG_DIR, f"{station_id}.flg"))
        except IOError:
            print(f"flag file missing for {station_id}")
            continue

        #*************************
        # read QFF
        # try:
        #     station_df = io.read(os.path.join(setup.SUBDAILY_OUT_DIR, f"{station_id}.qff"))
        # except IOError:
        #     print(f"Missing station {station_id}")
        #     continue

        for var in obs_var_list:

            setattr(station, var, utils.Meteorological_Variable(f"{var}", utils.MDI, "", ""))
            obs_var = getattr(station, var)

            # flags = station_df[f"{var}_QC_flag"].fillna("")

            for test in utils.QC_TESTS.keys():
                # locs = flags[flags.str.contains(test)]

                # setattr(obs_var, test, locs.shape[0]/flags.shape[0])
                # setattr(obs_var, f"{test}_counts", locs.shape[0])
                try:
                    setattr(obs_var, test, flag_summary[var][test])
                    setattr(obs_var, f"{test}_counts", flag_summary[var][f"{test}_counts"])
                except KeyError:
                    setattr(obs_var, test, 0)
                    setattr(obs_var, f"{test}_counts", 0)
                    

            # # for total, get number of clean obs and subtract
            # flagged, = np.where(flags != "")
            # setattr(obs_var, "All", flagged.shape[0]/flags.shape[0])
            # setattr(obs_var, "All_counts", flagged.shape[0])
            try:
                setattr(obs_var, "All", flag_summary[var]["All"])
                setattr(obs_var, "All_counts", flag_summary[var]["All_counts"])
                if diagnostics:
                    print(f"{var} - {flag_summary[var]['All_counts']}")
            except KeyError:
                setattr(obs_var, "All", 0)
                setattr(obs_var, "All_counts", 0)
                if diagnostics:
                    print(f"{var} - {0}")

        all_stations[station_id] = station

    # now spin through each var/test combo and make a plot
    for var in obs_var_list:
        for test in TESTS_FOR_VARS[var]:

            for suffix in ["", "_counts"]:

                lats, lons, flag_fraction = np.zeros(station_IDs.shape[0]), np.zeros(station_IDs.shape[0]), np.zeros(station_IDs.shape[0])

                for st, (ID, station) in enumerate(all_stations.items()):
                    lats[st] = station.lat
                    lons[st] = station.lon
                    obs_var = getattr(station, var)
                    flag_fraction[st] = getattr(obs_var, f"{test}{suffix}")


                if suffix == "":
                    flag_fraction *= 100.  # convert to percent

                # do the plot
                plt.figure(figsize=(8, 5))
                plt.clf()
                ax = plt.axes([0.02, 0.02, 0.96, 0.96], projection=ccrs.Robinson())
                ax.set_global()
                ax.coastlines('50m')

                try:
                    ax.gridlines()#draw_labels = True)
                except TypeError:
                    ax.gridlines()

                # colors are the exact same RBG codes as in IDL
                colors = [(150, 150, 150), (41, 10, 216), (63, 160, 255), (170, 247, 255), \
                          (255, 224, 153), (247, 109, 94), (165, 0, 33), (0, 0, 0)]
                if suffix == "":
                    limits = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 100.]
                elif suffix == "_counts":
                    limits = [0.0, 5., 10., 50., 100., 500., 1000., 5000.]
                    



                for u, upper in enumerate(limits):

                    # sort the labels
                    if u == 0:
                        locs, = np.where(flag_fraction == 0)
                        label = f"{upper}{UNITS[suffix]}: {len(locs)}"
                    else:
                        locs, = np.where(np.logical_and(flag_fraction <= upper, \
                                                        flag_fraction > limits[u-1]))
                        label = f">{limits[u-1]} to {upper}{UNITS[suffix]}: {len(locs)}"
                        if upper == limits[-1]:
                            label = f">{limits[u-1]}{UNITS[suffix]}: {len(locs)}"

                    # and plot
                    if len(locs) > 0:
                        ax.scatter(lons[locs], lats[locs], transform=ccrs.PlateCarree(), s=15, color=tuple([float(c)/255 for c in colors[u]]), \
                                   edgecolors="none", label = label)

                    else:
                        ax.scatter([0], [-90], transform=ccrs.PlateCarree(), s=15, color=tuple([float(c)/255 for c in colors[u]]), \
                                   edgecolors="none", label=label)

                if test == "All":
                    plt.title(f"{' '.join([v.capitalize() for v in var.split('_')])} - All")
                else:
                    plt.title(f"{' '.join([v.capitalize() for v in var.split('_')])} - {utils.QC_TESTS[test]}")

                watermarkstring="/".join(os.getcwd().split('/')[4:])+'/'+\
                    os.path.basename( __file__ )+"   "+dt.datetime.strftime(dt.datetime.now(), "%d-%b-%Y %H:%M")
                plt.figtext(0.01,0.01,watermarkstring,size=5)

                plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False, title='', prop={'size':9}, \
                               labelspacing=0.15, columnspacing=0.5, numpoints=1)

                print(os.path.join(IMAGE_LOCS, f"All_fails_{var}-{test}{suffix}_{start_time_string}.png"))
                plt.savefig(os.path.join(IMAGE_LOCS, f"All_fails_{var}-{test}{suffix}_{start_time_string}.png"))
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
