import numpy as np
import pandas as pd
from matplotlib import axes
import matplotlib.pyplot as plt

import io_utils as io
import utils
import setup

RETAINED_COLUMNS = ["datetime",
                    "temperature",
                    "temperature_QC_flag",
                    "dew_point_temperature",
                    "dew_point_temperature_QC_flag",
                    "sea_level_pressure",
                    "sea_level_pressure_QC_flag",
                    "station_level_pressure",
                    "station_level_pressure_QC_flag",
                    "wind_speed",
                    "wind_speed_QC_flag",
                    "wind_direction",
                    "wind_direction_QC_flag"]

VARS = ["temperature",
        "dew_point_temperature",
        "sea_level_pressure",
        "station_level_pressure",
        "wind_speed",
        "wind_direction",
        ]

FLAG_COLOURS = {"C" : "b",
                "D" : "b",
                "E" : "b",
                "F" : "b",
                "H" : "b",
                "K" : "c",
                "L" : "c",
                "N" : "c",
                "S" : "c",
                "T" : "c",
                "U" : "m",
                "V" : "m",
                "W" : "m",
                "d" : "m",
                "h" : "m",
                "n" : "limegreen",
                "o" : "limegreen",
                "p" : "limegreen",
                "w" : "limegreen",
                "x" : "limegreen",
                "y" : "orange",
                "1" : "orange",
                "2" : "orange",}

FLAG_LOCS = {"C" : 0.1,
            "D" : 0.3,
            "E" : 0.5,
            "F" : 0.7,
            "H" : 0.9,
            "K" : 0.1,
            "L" : 0.3,
            "N" : 0.5,
            "S" : 0.7,
            "T" : 0.9,
            "U" : 0.1,
            "V" : 0.3,
            "W" : 0.5,
            "d" : 0.7,
            "h" : 0.9,
            "n" : 0.1,
            "o" : 0.3,
            "p" : 0.5,
            "w" : 0.7,
            "x" : 0.9,
            "y" : 0.1,
            "1" : 0.3,
            "2" : 0.5,}

assert len(VARS) == len(setup.obs_var_list)
assert 2 * len(VARS) + 1 == len(RETAINED_COLUMNS)


def plot_data_and_flags(ax: axes.Axes,
                        df: pd.DataFrame,
                        varname: str) -> None:
    """Plot the data and flags

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axes instance on which to plot
    df : pd.DataFrame
        DataFrame containing data
    varname : str
        Variable to pull out and plot
    """
    transform = ax.get_xaxis_transform()
    # plot data
    ax.plot(df.index, df[varname], "ko", ms=5)

    # pull out rows where flags are set
    flag_df = df[df[f"{varname}_QC_flag"].notnull()]

    # spin through the flags
    for key, value in FLAG_COLOURS.items():

        this_flag_df_locs = flag_df[f"{varname}_QC_flag"].str.contains(key)
        this_flag_df = flag_df[this_flag_df_locs]

        # plot these in axes coordinates, so that they are separate from the points
        ax.plot(this_flag_df.index, [FLAG_LOCS[key] for i in range(this_flag_df.shape[0])],
                c=value, marker="|", ms=10, ls="", transform=transform)

        # and plot the flagged values so they stand out too
        ax.plot(this_flag_df.index, this_flag_df[varname],
                c="r", marker="o", ms=5, ls="",)
        # set the ylabel
    ax.set_ylabel(" ".join([i.capitalize() for i in varname.split("_")]))


def plot_year(df: pd.DataFrame, year: int,
              save: bool=False) -> None:
    """Plot a single year from the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data to plot
    year : int
        Which year to plot
    """

    fig, axes = plt.subplots(6, 1, sharex=True,
                             gridspec_kw=dict(hspace=0),
                             figsize=(8, 20))

    for ax, var in zip(axes, VARS):

        plot_data_and_flags(ax, df, var)

    fig.tight_layout(h_pad=0)

    fig.suptitle(str(year))
    if save:
        plt.savefig(f"{year}.png")
    else:
        plt.show()
    plt.close()


def main(station_id: str,
         save: bool) -> None:
    # process the station list
    station_list = utils.get_station_list()
    station_IDs = station_list.id
    this_station = station_IDs[station_IDs == station_id].index[0]


    station = utils.Station(station_id, station_list.latitude[this_station],
                            station_list.longitude[this_station],
                            station_list.elevation[this_station])

    # TODO: to set this as a selectable choice
    if "internal" =="internal":
        station, station_df = io.read_station(setup.SUBDAILY_PROC_DIR /
                                                "{:11s}{}{}".format(station_id,
                                                                    setup.OUT_SUFFIX,
                                                                    setup.OUT_COMPRESSION),
                                                station)
    else:
        station, station_df = io.read_station(setup.SUBDAILY_OUT_DIR /
                                                "{:11s}{}{}".format(station_id,
                                                                    setup.OUT_SUFFIX,
                                                                    setup.OUT_COMPRESSION),
                                                station)

    # add datetime column for plotting
    datetimes = io.calculate_datetimes(station_df)
    station_df["datetime"] = datetimes

    all_years = np.unique(station.years)

    for year in all_years:

        this_year_df = station_df[station_df["Year"] == year][RETAINED_COLUMNS]
        # remove empty rows
        this_year_df.dropna(how="all", inplace=True)
        this_year_df.set_index("datetime", inplace=True)

        if this_year_df.shape[0] == 0:
            continue

        plot_year(this_year_df, year, save=save)






#************************************************************************
if __name__ == "__main__":

    import argparse

    # set up keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--station_id', dest='station_id', action='store', default="",
                        help='Station ID to plot, default=""')
    parser.add_argument('--save', dest='save', action='store_true', default=False,
                        help='Save output files, default is for interactive')
    

    args = parser.parse_args()

    if args.station_id == "":
        print("Please enter ID to plot")
    else:
        main(station_id=args.station_id, save=args.save)

#*******************************************************
