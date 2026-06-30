import numpy as np

import utils


def clc(station: utils.Station, config_dict: dict, full: bool=False,
        plots: bool=False, diagnostics: bool=False) -> None:
    """Cloud Logical Check

    Parameters
    ----------
    station : utils.Station
        Station data
    config_dict : dict
        dictionary for configuration sections
    full : bool, optional
        run a full update, by default False
    plots : bool, optional
        turn on plotting, by default False
    diagnostics : bool, optional
        turn on diagnostic outputs, by default False
    """

    # sky_cover_layer_1/2/3/4
    # sky_cover_layer_baseht_1/2/3/4

    # need more than one measured layer to find inconsistencies
    heights = np.ma.vstack((station.sky_cover_layer_baseht_1.data,
                            station.sky_cover_layer_baseht_2.data,
                            station.sky_cover_layer_baseht_3.data,
                            station.sky_cover_layer_baseht_4.data,))

    obs_counts = heights.count(axis=0)
    locs, = np.nonzero(obs_counts > 1)

    # account for cases where heights from fields 1 to 4 aren't in numerical order
    sort_order = np.ma.argsort(heights[:, locs], axis=0)

    # now can check the cloud fields
    oktas = np.ma.vstack((station.sky_cover_layer_1.data,
                          station.sky_cover_layer_2.data,
                          station.sky_cover_layer_3.data,
                          station.sky_cover_layer_4.data,))

    # Now need to find locations where have values at layer above one with 8oktas

    input("stop")

