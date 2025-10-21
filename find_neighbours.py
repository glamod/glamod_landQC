'''
Find Neighbours
===============

Find the neighbours for each station and store in file.  Needs to be run before the neighbour
checks (``inter_checks.py``) and then each time the station list is updated.

find_neighbours.py invoked by typing::

  python find_neighbour.py --restart_id --end_id [--full] [--plots] [--diagnostics]

Input arguments:

--restart_id        First station to process

--end_id            Last station to process

--full              [False] Run a full reprocessing (recalculating thresholds) rather than reading from files

--plots             [False] Create plots (maybe interactive)

--diagnostics       [False] Verbose output

'''

DEFAULT_SPHERICAL_EARTH_RADIUS=6367470

import numpy as np
import pandas as pd

from geometry import polar2d_to_cartesian, cross_distance

import utils
from setup import SUBDAILY_CONFIG_DIR

DEFAULT_SEPARATION = 9999
CHUNKSIZE = 1000 # size of arrays to split ID list into for distances

#************************************************************************
def get_cartesian(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Compute the matrix of coranges between all location in a grid
    Args:
        latitudes:
            A 1D array of M locations on the latitude coordinate.
        longitudes:
            A 1D array of N locations on the longitudes coordinate.

    Returns:
        cartesian_coords:
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z] on the
            surface of a unit sphere corresponding to the latitude, longitude pairs in polar2d.
    """
    polar2d_coords = np.vstack((latitudes, longitudes)).T

    cartesian_coords = polar2d_to_cartesian(polar2d_coords)

    return cartesian_coords # get_cartesian

#************************************************************************
def compute_corange_matrix(cartesian_coords_a: np.ndarray, cartesian_coords_b: np.ndarray = None) -> np.ndarray:
    """Compute the matrix of coranges between all location in a grid
    Args:
        cartesian_coords_a:
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z] on the
            surface of a unit sphere corresponding to the latitude, longitude pairs in polar2d.
    Kwargs
        cartesian_coords_b:
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z] on the
            surface of a unit sphere corresponding to the latitude, longitude pairs in polar2d.
    Returns:
        An (M*N, M*N) array of great circle distances between all locations.
    """

    # Compute cross distance matrix
    coranges = cross_distance(cartesian_coords_a, locations_b=cartesian_coords_b, R=DEFAULT_SPHERICAL_EARTH_RADIUS / 1000.)

    return coranges # compute_corange_matrix

#************************************************************************
def get_neighbours(station_list_a: pd.DataFrame, station_list_b: pd.DataFrame = None,
                   diagnostics: bool = False, plots: bool = False, full: bool = False) -> np.ndarray:
    """
    Find the neighbour indices and distances for the list supplied

    :param dataframe station_list_a: id, latitude, longitude, elevation and name.
    :param dataframe station_list_b: id, latitude, longitude, elevation and name [None].
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots
    :param bool full: run full reprocessing rather than using stored values.

    :returns: array of [targets, neighbour_indices, distances]
    """


    # storage array for index and distance [index, neighbour_index, distance]
    neighbours = -np.ones((station_list_a.shape[0], utils.MAX_N_NEIGHBOURS, 2)).astype(int)
    neighbours[:, :, 1] = DEFAULT_SEPARATION

    if diagnostics:
        print("Finding distances")

    # distances are important, bearings are subsidiary - can be calculated afterwards
    cartesian_a = get_cartesian(station_list_a.latitude, station_list_a.longitude)
    if station_list_b is not None:
        cartesian_b = get_cartesian(station_list_b.latitude, station_list_b.longitude)
    else:
        cartesian_b = None
    distances = compute_corange_matrix(cartesian_a, cartesian_b).astype(int)

    # and do elevation check
    elev_a = station_list_a.elevation
    if station_list_b is not None:
        elev_b = station_list_b.elevation
    else:
        elev_b = np.copy(station_list_a.elevation)
    elev_a, elev_b = np.meshgrid(elev_a, elev_b)
    vertical_separations = np.abs(elev_a - elev_b).T # transpose to match shape of distances

    if diagnostics:
        print("Post-processing")

    # store indices of those if close enough.
    for t, target in enumerate(distances):

        match_distance, = np.where(target <= utils.MAX_NEIGHBOUR_DISTANCE)
        match_elevation, = np.where(vertical_separations[t] <= utils.MAX_NEIGHBOUR_VERTICAL_SEP)
        matches = np.intersect1d(match_distance, match_elevation)

        if len(matches) > 0:
            # have some neighbours

            current_neighbours = neighbours[t, :, :]

            # append to 2-d array
            new_neighbours = np.array((matches, target[matches])).T
            current_neighbours = np.append(current_neighbours, new_neighbours, axis=0)

            # sort on distances, and store closest N
            sort_order = np.argsort(current_neighbours[:, 1])

            # overwrite.
            neighbours[t, :, :] = current_neighbours[sort_order[:utils.MAX_N_NEIGHBOURS]]

    return neighbours # get_neighbours

#************************************************************************
def main(restart_id: str = "", end_id: str = "", diagnostics: bool = False, plots: bool = False, full: bool = False) -> None:
    """
    Find all possible neighbours.  Works in chunks to save on resources.

    :param str restart_id: which station to start on
    :param str end_id: which station to end on
    :param bool diagnostics: print extra material to screen
    :param bool plots: create plots
    :param bool full: run full reprocessing rather than using stored values.
    """

    station_list = utils.get_station_list(restart_id=restart_id, end_id=end_id)

    # storage array for index and distance [index, neighbour_index, distance]
    neighbours = -np.ones((station_list.shape[0], utils.MAX_N_NEIGHBOURS, 2)).astype(int)
    neighbours[:, :, 1] = DEFAULT_SEPARATION

    # now need to chunk up and process in bits.
    if station_list.shape[0] <= CHUNKSIZE:
        # this avoids an error that otherwise gets thrown by the "//" in below
        #  likely only during testing situations
        sub_arrays = [station_list]
    else:
        sub_arrays = np.array_split(station_list, station_list.shape[0]//CHUNKSIZE)

    # process sub-arrays down
    for sa1, sub_arr1 in enumerate(sub_arrays):
        if diagnostics:
            print(f"{sa1+1}/{len(sub_arrays)}")
        # extract neighbour array to work on
        these_station_neighbours = neighbours[sub_arr1.index.start : sub_arr1.index.stop]

        # process sub-arrays across
        for sub_arr2 in sub_arrays:
            these_neighbours = get_neighbours(sub_arr1, sub_arr2, diagnostics=diagnostics)

            # adjust set arrayindices to account for chunking
            indices = these_neighbours[:, :, 0]
            indices[indices != -1] += sub_arr2.index.start
            these_neighbours[:, :, 0] = indices

            these_station_neighbours = np.append(these_station_neighbours, these_neighbours, axis=1) # append on neighbours-axis

            # doing longhand as couldn't get a reliable whole-array solution
            for sn, this_station in enumerate(these_station_neighbours):
                # sort on distances, and store closest N
                sort_order = np.argsort(this_station[:, 1])

                # overwrite.
                these_station_neighbours[sn] = this_station[sort_order]

        # write back into neighbour array
        neighbours[sub_arr1.index.start : sub_arr1.index.stop] = these_station_neighbours[:, :utils.MAX_N_NEIGHBOURS, :]

    # so this only needs running once per update, write out and store the neighbours
    if diagnostics:
        print("writing")
    with open(utils.NEIGHBOUR_FILE, "w") as outfile:

        # each station
        for st, station in enumerate(neighbours):
            """In cases where stations with lat=0 and lon=0 is pervasive, there could be
            more than MAX_N_NEIGHBOURS with zero distance.  Hence the sorting by distance
            won't necessarily end up with the target station at the first index location.
            These stations will be withheld by the logic checks, so no buddy checks will
            be run.  Hence, can manually overwrite the first entry to ensure the writing works."""

            if station_list.latitude[st] == 0 and station_list.longitude[st] == 0:
                # this station should be withheld by the logic checks, so no buddy checks will be run
                zeros, = np.where(station[:, 1] == 0)
                if len(zeros) == len(station[:, 1]):
                    # checking all neighbours have zero distance
                    if station[0, 0] != st and station[0, 1] == 0:
                        # can just overwrite the first
                        station[0, 0] = st

            # check to make sure that the first entry is correct (in cases where a neighbour has zero separation
            if station[0, 0] != st:
                zeros, = np.where(station[:, 1] == 0)
                match, = np.where(station[zeros, 0] == st)
                if len(match) == 1:
                    station[[0, match[0]]]=station[[match[0], 0]]
                else:
                    input(f"{station} has issue")

            outstring = ""
            # each neighbour
            for neighb in station:
                if neighb[0] != -1:
                    outstring = f"{outstring:s} {station_list.id[neighb[0]]:<11s} {neighb[1]:8d}"
                else:
                    outstring = f"{outstring:s} {'-':>11s} {neighb[1]:8d}"

#            input("stop")

            outfile.write(f"{outstring}\n")

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
    parser.add_argument('--full', dest='full', action='store_true', default=False,
                        help='Run full reprocessing rather than just an update')
    parser.add_argument('--diagnostics', dest='diagnostics', action='store_true', default=False,
                        help='Run diagnostics (will not write out file)')
    parser.add_argument('--plots', dest='plots', action='store_true', default=False,
                        help='Run plots (will not write out file)')
    args = parser.parse_args()

    main(restart_id=args.restart_id,
               end_id=args.end_id,
               diagnostics=args.diagnostics,
               plots=args.plots,
               full=args.full)


