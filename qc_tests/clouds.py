import numpy as np

import utils


def get_heights_and_oktas(station: utils.Station) -> tuple[np.ma.MaskedArray,
                                                           np.ma.MaskedArray]:
    """Extract cloud level heights and values from the station object

    Parameters
    ----------
    station : utils.Station
        Station object holding the data

    Returns
    -------
    tuple[np.ma.MaskedArray, np.ma.MaskedArray]
        Masked Arrays of heights and oktas
    """

    # need more than one measured layer to find inconsistencies
    heights = np.ma.vstack((station.sky_cover_layer_baseht_1.data,
                            station.sky_cover_layer_baseht_2.data,
                            station.sky_cover_layer_baseht_3.data,
                            station.sky_cover_layer_baseht_4.data,))

    # now can check the cloud fields
    oktas = np.ma.vstack((station.sky_cover_layer_1.data,
                          station.sky_cover_layer_2.data,
                          station.sky_cover_layer_3.data,
                          station.sky_cover_layer_4.data,))

    return heights, oktas


def orphan_values(heights, oktas, hflags, oflags) -> None:
    # test for heights without oktas and vice versa

    # Check for combination of:
    #    - where the mask for heights and oktas differ
    #    - and either:
    #          - oktas <= 8 [as values of 9 & 10 mean (partial) sky
    #                        obscuration hence heights would not be
    #                        determined in this case]
    #              or:
    #          - oktas mask is True [so cannot test value]
    suspect_locs = np.nonzero(np.logical_and(heights.mask != oktas.mask,
                                             np.logical_or(oktas.data <= 8,
                                                           oktas.mask==True)))

    hflags[suspect_locs] = 1
    oflags[suspect_locs] = 1


def obscured_heights(heights, oktas, hflags) -> None:

    # Check for heights when have obscuration
    suspect_obscured_locs = np.nonzero(np.logical_and(oktas.data >= 9,
                                                      heights.mask==False))

    hflags[suspect_obscured_locs] = 1


    # TODO: add logging/diagnostic info


def process_erroneous_clouds(oktas, hflags, oflags):

    # Now need to find locations where have values at layer above one with 8 oktas
    timestamp, full_layer = np.nonzero(oktas == 8)

    # checking each full-cloud timestamp in sequence
    for tt, ll in zip(timestamp, full_layer):
        flags = np.zeros(4)

        # find values at heights above the 8-okta level
        above_full = oktas[tt, ll:]

        if len(above_full.compressed()) == 1:
            # If only have the 8 okta value and no measurements
            #    above that, then all fine
            continue

        # If there measured cloud values above that, then need to
        #    set the flags on a temporary array
        flags[ll+1:] = 1
        # retain the masks when copying over
        hflags[tt, :] = np.ma.array(flags, mask=hflags.mask[tt])
        oflags[tt, :] = np.ma.array(flags, mask=oflags.mask[tt])



def process_multiple_layers(heights, oktas,
                            hflags, oflags) -> tuple[np.ma.masked_array,
                                                     np.ma.masked_array]:


    # account for cases where heights from fields 1 to 4 aren't in numerical order
    sort_order = np.ma.argsort(heights, axis=1)

    # Sort the cloud amounts and flags following height order
    sorted_oktas = np.take_along_axis(oktas, sort_order, axis=1)
    sorted_oflags = np.take_along_axis(oflags, sort_order, axis=1)
    sorted_hflags = np.take_along_axis(hflags, sort_order, axis=1)

    # process in child process so can work on views
    process_erroneous_clouds(sorted_oktas,
                             sorted_hflags,
                             sorted_oflags)

    # and unpack back to correct locations
    np.put_along_axis(oflags, sort_order, sorted_oflags, axis=1)
    np.put_along_axis(hflags, sort_order, sorted_hflags, axis=1)

    return hflags, oflags



def logical_cross_check(heights, oktas, hflags, oflags) -> None:

    # Only necessary to check in places where there are measurements at
    #   more than one level
    obs_counts = heights.count(axis=1)
    multiple_layers, = np.nonzero(obs_counts > 1)

    (hflags[multiple_layers],
    oflags[multiple_layers]) = process_multiple_layers(heights[multiple_layers],
                                                      oktas[multiple_layers],
                                                      hflags[multiple_layers],
                                                      oflags[multiple_layers])



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

    heights, oktas = get_heights_and_oktas(station)

    # set up the flag array
    okta_flags = np.ma.zeros(oktas.shape)
    okta_flags.mask = oktas.mask[:]
    height_flags = np.ma.zeros(heights.shape)
    height_flags.mask = oktas.mask[:]

    orphan_values(heights, oktas, height_flags, okta_flags)

    obscured_heights(heights, oktas, height_flags)

    logical_cross_check(heights, oktas, height_flags, okta_flags)

