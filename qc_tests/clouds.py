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


def orphan_values(heights: np.ma.MaskedArray,
                  oktas: np.ma.MaskedArray,
                  hflags: np.ma.MaskedArray,
                  oflags: np.ma.MaskedArray) -> None:
    """
    Test for heights without oktas and vice versa

    Check for combination of:
       - where the mask for heights and oktas differ
       - and either:
             - oktas <= 8 [as values of 9 & 10 mean (partial) sky
                           obscuration hence heights would not be
                           determined in this case]
                 or:
             - oktas mask is True [so cannot test value]

    Parameters
    ----------
    heights : np.ma.MaskedArray
        Masked array of cloud heights for 4 layers (time x layer)
    oktas : np.ma.MaskedArray
        Masked array of cloud oktas for 4 layers (time x layer)
    hflags : np.ma.MaskedArray
        Masked array of cloud height flags for 4 layers (time x layer)
    oflags : np.ma.MaskedArray
        Masked array of cloud okta flags for 4 layers (time x layer)
    """

    suspect_locs = np.nonzero(np.logical_and(heights.mask != oktas.mask,
                                             np.logical_or(oktas.data <= 8,
                                                           oktas.mask==True)))

    hflags[suspect_locs] = 1
    oflags[suspect_locs] = 1


def obscured_heights(heights: np.ma.MaskedArray,
                     oktas: np.ma.MaskedArray,
                     hflags: np.ma.MaskedArray) -> None:
    """
    Check for height values where oktas indicate (partial) obscuration.

    When oktas = 9 or 10, these arise from the following:

        VV:09 = Sky obscured, or cloud amount cannot be estimated
        X:10 = Partial obscuration

    Flag instances in place

    Parameters
    ----------
    heights : np.ma.MaskedArray
        Masked array of cloud heights for 4 layers (time x layer)
    oktas : np.ma.MaskedArray
        Masked array of cloud oktas for 4 layers (time x layer)
    hflags : np.ma.MaskedArray
        Masked array of cloud height flags for 4 layers (time x layer)
    """

    # Check for heights when have obscuration
    suspect_obscured_locs = np.nonzero(np.logical_and(oktas.data >= 9,
                                                      heights.mask==False))

    hflags[suspect_obscured_locs] = 1


    # TODO: add logging/diagnostic info


def process_erroneous_clouds(oktas: np.ma.MaskedArray,
                             hflags: np.ma.MaskedArray,
                             oflags: np.ma.MaskedArray) -> None:
    """With okta information from multiple layers in height
    order, identify locations where measurements have been made
    above (to the right) of okta=8 full cloud.

    Parameters
    ----------
    oktas : np.ma.MaskedArray
        Masked array of cloud oktas for 4 layers (time x layer)
    hflags : np.ma.MaskedArray
        Masked array of cloud height flags for 4 layers (time x layer)
    oflags : np.ma.MaskedArray
        Masked array of cloud okta flags for 4 layers (time x layer)

    """
    # First find those timestamps which have a full layer somewhere
    timestamp, full_layer = np.nonzero(oktas == 8)

    # Checking each full-cloud timestamp in sequence
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
    """Taking those timestamps where there are cloud
    measurements at more than one layer, need to sort into
    height order before doing the logical check.

    Because of 2d nature of array, use `take_along_axis` to sort
    and then `put_along_axis` to unsort a new array.

    Parameters
    ----------
    heights : np.ma.MaskedArray
        Masked array of cloud heights for 4 layers (time x layer)
    oktas : np.ma.MaskedArray
        Masked array of cloud oktas for 4 layers (time x layer)
    hflags : np.ma.MaskedArray
        Masked array of cloud height flags for 4 layers (time x layer)
    oflags : np.ma.MaskedArray
        Masked array of cloud okta flags for 4 layers (time x layer)

    Returns
    -------
    tuple[np.ma.MaskedArray, np.ma.MaskedArray] : height and okta flags respectively
    """
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


def logical_cross_check(heights: np.ma.MaskedArray,
                        oktas: np.ma.MaskedArray,
                        hflags: np.ma.MaskedArray,
                        oflags: np.ma.MaskedArray) -> None:
    """Top level routine for the cloud logical check.
    Want to check that there are no values at heights above
    that where oktas=8 (overcast, full cloud).

    Here, we select only those timestamps which have more than
    one layer.  Using nested routines because we need to keep
    the flags linked correctly to the layers[1-4].

    Parameters
    ----------
    heights : np.ma.MaskedArray
        Masked array of cloud heights for 4 layers (time x layer)
    oktas : np.ma.MaskedArray
        Masked array of cloud oktas for 4 layers (time x layer)
    hflags : np.ma.MaskedArray
        Masked array of cloud height flags for 4 layers (time x layer)
    oflags : np.ma.MaskedArray
        Masked array of cloud okta flags for 4 layers (time x layer)
    """

    # Only necessary to check in places where there are measurements at
    #   more than one level
    obs_counts = heights.count(axis=1)
    multiple_layers, = np.nonzero(obs_counts > 1)

    (hflags[multiple_layers],
    oflags[multiple_layers]) = process_multiple_layers(heights[multiple_layers],
                                                      oktas[multiple_layers],
                                                      hflags[multiple_layers],
                                                      oflags[multiple_layers])


def insert_cloud_flags(station: utils.Station,
                       variable: "str",
                       binary_flags: np.ma.MaskedArray) -> None:
    """Flags have been set on a binary array (1/0) but
    will need to be changed to a string array, so do this
    for the variable given in the kwargs.

    Parameters
    ----------
    station : utils.Station
        Station object to update
    variable : str
        Which variable to process
    binary_flags : np.ma.MaskedArray
        Flag array in binary 1/0 flagged/unflagged.
    """
    # make the character flag array, and copy over flags
    character_flags = np.array(["" for i in range(binary_flags.shape[0])])
    character_flags[binary_flags == 1] = utils.QC_TEST_FLAGS["Clouds"]

    # get the variablel and store the flags
    this_variable = getattr(station, variable)
    this_variable.store_flags(utils.insert_flags(this_variable.flags,
                                                 character_flags))


def clc(station: utils.Station, config_dict: dict, full: bool=False,
        plots: bool=False, diagnostics: bool=False) -> None:
    """Cloud Logical Checks

    Looking for logical inconsistency in cloud layer information

    This set of tests processes the following variables:
      sky_cover_layer_1/2/3/4
      sky_cover_layer_baseht_1/2/3/4

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

    # read into an array of (Time x Layer)
    heights, oktas = get_heights_and_oktas(station)

    # set up the flag array
    okta_flags = np.ma.zeros(oktas.shape)
    okta_flags.mask = oktas.mask[:]
    height_flags = np.ma.zeros(heights.shape)
    height_flags.mask = oktas.mask[:]

    # heights without cloud values, and vice versa
    orphan_values(heights, oktas, height_flags, okta_flags)

    # heights when cloud values says obscured
    obscured_heights(heights, oktas, height_flags)

    # any cloud values above an okta=8 reading
    logical_cross_check(heights, oktas, height_flags, okta_flags)

    # need to insert flags
    for v, var in enumerate("sky_cover_layer_1",
                            "sky_cover_layer_2",
                            "sky_cover_layer_3",
                            "sky_cover_layer_4"):
        insert_cloud_flags(station, var, okta_flags[:, v])

    for v, var in enumerate("sky_cover_layer_baseht_1",
                            "sky_cover_layer_baseht_2",
                            "sky_cover_layer_baseht_3",
                            "sky_cover_layer_baseht_4"):
        insert_cloud_flags(station, var, height_flags[:, v])
