"""
Contains tests for clouds.py
"""
import numpy as np
from unittest.mock import patch, Mock

import clouds

import common


def test_get_heights_and_oktas() -> None:
    """Check that processing of height and okta information
    into arrays happens as intended"""

    layer_1 = common.example_test_variable("sky_cover_layer_1", np.ma.arange(5))
    station = common.example_test_station(layer_1)

    # add remaining fields
    station.sky_cover_layer_2 = common.example_test_variable("sky_cover_layer_2",
                                                             np.ma.arange(5)+1)
    station.sky_cover_layer_3 = common.example_test_variable("sky_cover_layer_3",
                                                             np.ma.arange(5)+2)
    station.sky_cover_layer_4 = common.example_test_variable("sky_cover_layer_4",
                                                             np.ma.arange(5)+3)

    station.sky_cover_layer_baseht_1 = common.example_test_variable(
        "sky_cover_layer_baseht_1",
        np.ma.arange(5)*10)
    station.sky_cover_layer_baseht_2 = common.example_test_variable(
        "sky_cover_layer_baseht_2",
        np.ma.arange(5)*20)
    station.sky_cover_layer_baseht_3 = common.example_test_variable(
        "sky_cover_layer_baseht_3",
        np.ma.arange(5)*30)
    station.sky_cover_layer_baseht_4 = common.example_test_variable(
        "sky_cover_layer_baseht_4",
        np.ma.arange(5)*40)

    heights, oktas = clouds.get_heights_and_oktas(station)

    # in this test only ensuring structure,
    #   5 rows of timestamps, 4 columns of layers. Asymmetric
    expected_oktas = np.ma.array([[0, 1, 2, 3],
                                  [1, 2, 3, 4],
                                  [2, 3, 4, 5],
                                  [3, 4, 5, 6],
                                  [4, 5, 6, 7]])
    expected_heights = np.ma.array([[0, 0, 0, 0],
                                    [10, 20, 30, 40],
                                    [20, 40, 60, 80],
                                    [30, 60, 90, 120],
                                    [40, 80, 120, 160]])

    np.testing.assert_array_equal(expected_heights, heights)
    np.testing.assert_array_equal(expected_oktas, oktas)


def test_orphan_values() -> None:
    """Check to see that test identifies heights or okta values
    which do not have a corresponding value in the other quantity"""

    # set up example arrays (5 timestamps, 4 layers)
    heights = np.array([[50, 100, 150, 200],   # normal
                        [50, -100, 150, 200],  # no corresponding mask in oktas
                        [50, 100, 150, 200],
                        [50, 100, 150, -200],
                        [50, 100, 150, -200]])
    oktas = np.array([[1, 2, 3, 4],
                      [1, 2, 3, 4],
                      [1, 2, -1, 4],  # no corresponding mask in heights
                      [1, 2, 3, 9],   # 9 is valid and reflected in heights
                      [1, 2, 3, 10]])  # 10 is valid and reflected in heights

    heights = np.ma.masked_where(heights < 0, heights)
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.zeros(heights.shape)
    oflags = np.zeros(oktas.shape)

    clouds.orphan_values(heights, oktas, hflags, oflags)

    expected_hflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])
    expected_hflags.mask = heights.mask

    expected_oflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])
    expected_oflags.mask = oktas.mask

    np.testing.assert_array_equal(hflags, expected_hflags)
    np.testing.assert_array_equal(oflags, expected_oflags)


def test_obscured_heights() ->  None:
    """Check to see if routine identify height values
    which should not be possible given obscured okta code"""

    # set up example arrays (3 timestamps, 4 layers)
    heights = np.array([[50, 100, 150, -200],
                        [50, 100, 150, 200],
                        [50, 100, 150, -200]])
    oktas = np.array([[1, 2, 3, 9],
                      [1, 2, 3, 9],   # 9 is valid and not reflected in heights
                      [1, 2, 3, 10]])

    heights = np.ma.masked_where(heights < 0, heights)
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.zeros(heights.shape)

    clouds.obscured_heights(heights, oktas, hflags)

    expected_hflags = np.ma.array([[0, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 0, 0]])
    expected_hflags.mask = heights.mask

    np.testing.assert_array_equal(hflags, expected_hflags)


def test_process_erroneous_clouds() -> None:
    """Check to see that test picks up cloud measurements
    above a full sky (okta=8) value.

    This routine expects height ordered data, so looking for
    values to the right of an 8"""

    # set up example arrays (5 timestamps, 4 layers)
    oktas = np.array([[1, 2, 8, -1], # OK
                      [1, 2, 8, 4], # bad
                      [1, 2, 3, 4], # OK
                      [0, 8, 1, -1], # bad
                      [1, 8, 2, -1]]) # bad
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.ma.zeros(oktas.shape)
    hflags.mask = oktas.mask
    oflags = np.ma.zeros(oktas.shape)
    oflags.mask = oktas.mask

    clouds.process_erroneous_clouds(oktas, hflags, oflags)

    expected_hflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 1],
                                   [0, 0, 1, 0]])
    expected_hflags.mask = oktas.mask

    expected_oflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0],
                                   [0, 0, 1, 1],
                                   [0, 0, 1, 0]])
    expected_oflags.mask = oktas.mask

    np.testing.assert_array_equal(expected_oflags, oflags)
    np.testing.assert_array_equal(expected_hflags, hflags)


@patch("clouds.process_erroneous_clouds")
def test_process_multiple_layers_calls(process_clouds_mock: Mock) -> None:
    """To test the functionality of this routine, need to see what the child
    is called with, hence the mocking.

    The okta values are ordered by the height values, and then
    used to call, so check these are done corretly
    """
    # set up example arrays (3 timestamps, 4 layers)
    oktas = np.array([[1, 2, 3, 4],
                      [1, 2, 3, 4],
                      [1, 2, -1, 4]]) # to test masked values, but Oktas OK

    heights = np.array([[50, 100, 150, 200],
                        [50, 100, 200, 150], # wrong order
                        [50, 200, -1, 100]]) # wrong order and with mask
    heights = np.ma.masked_where(heights < 0, heights)
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.zeros(heights.shape)
    oflags = np.zeros(oktas.shape)

    clouds.process_multiple_layers(heights, oktas, hflags, oflags)

    # this is what we expect the call to look like, oktas
    #    rearranged in height order
    expected_oktas = np.array([[1, 2, 3, 4],
                               [1, 2, 4, 3],
                               [1, 4, -1, 2]])
    expected_oktas = np.ma.masked_where(expected_oktas < 0,
                                        expected_oktas)

    calls = process_clouds_mock.call_args_list[0]
    np.testing.assert_array_equal(calls.args[0],
                                  expected_oktas)


@patch("clouds.process_multiple_layers")
def test_logical_cross_check_calls(proc_layers_mock: Mock) -> None:
    """To test the functionality of this routine, need to see
    what the child is called with, hence the mocking.

    Test that only timestamps with multiple cloud layers are
    passed into child"""

    # set up example arrays
    oktas = np.array([[1, -1, -1, -1], #  1 layer only
                      [1, 2, 3, -1]])   #  more than 1 layer
    heights = np.array([[50, -1, -1, -1],
                        [50, 100, 150, -1]])
    heights = np.ma.masked_where(heights < 0, heights)
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.zeros(heights.shape)
    oflags = np.zeros(oktas.shape)

    proc_layers_mock.return_value = (np.zeros((1,4)), np.zeros((1,4)))

    clouds.logical_cross_check(heights, oktas, hflags, oflags)

    proc_layers_mock.assert_called_once()

    # only expect second timestamp to be passed into child as this has
    #    more than 1 layer. Check call arguments
    calls = proc_layers_mock.call_args_list[0]
    np.testing.assert_array_equal(calls.args[0],
                                  heights[[1]])
    np.testing.assert_array_equal(calls.args[1],
                                  oktas[[1]])
    np.testing.assert_array_equal(calls.args[2],
                                  hflags[[1]])
    np.testing.assert_array_equal(calls.args[3],
                                  oflags[[1]])


def test_logical_cross_check() -> None:
    """Full test of logical cross check in clouds to
    ensure that flags set correctly."""

    # set up example arrays (5 timestamps, 4 layers)
    oktas = np.array([[1, -1, -1, -1], #  1 layer only
                      [1, 2, 3, -1],   #  more than 1 layer
                      [1, 2, 8, -1],   #  full but no measurements above
                      [1, 2, 8, 3],   # should trigger flag
                      [1, 2, 8, 3]])  # heights in non numerical, but reordered oktas OK

    heights = np.array([[50, -1, -1, -1],
                        [50, 100, 150, -1],
                        [50, 100, 150, 200],
                        [50, 100, 150, 200],
                        [50, 100, 200, 150]])
    heights = np.ma.masked_where(heights < 0, heights)
    oktas = np.ma.masked_where(oktas < 0, oktas)

    hflags = np.ma.zeros(heights.shape)
    hflags.mask = heights.mask
    oflags = np.ma.zeros(oktas.shape)
    oflags.mask = oktas.mask

    clouds.logical_cross_check(heights, oktas, hflags, oflags)

    expected_hflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]])
    expected_hflags.mask = heights.mask

    expected_oflags = np.ma.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]])
    expected_oflags.mask = oktas.mask

    np.testing.assert_array_equal(hflags, expected_hflags)
    np.testing.assert_array_equal(oflags, expected_oflags)


def test_insert_cloud_flags() -> None:
    """Test to ensure that flags being stored as expected"""

    layer_1 = common.example_test_variable("sky_cover_layer_1", np.ma.arange(5))
    layer_1.store_flags(np.array(["", "", "", "", ""]))
    station = common.example_test_station(layer_1)

    flags = np.array([0, 1, 0, 1, 0])
    clouds.insert_cloud_flags(station, "sky_cover_layer_1", flags)

    np.testing.assert_array_equal(station.sky_cover_layer_1.flags,
                                  np.array(["", "y", "", "y", ""]))
