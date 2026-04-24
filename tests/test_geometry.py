# (C) British Crown Copyright 2026, Met Office.
# Please see LICENSE for license details.

'''
This module contains tests for :mod:`geometry.py`.

Taken from HadCRUT analysis tests
'''
import pytest
import numpy as np

import geometry


class Dummy(object):
    """HadCRUT approach used unittest and a class,
    so lifting this here"""

    def __init__(self):

        # matching locations in xyz and llr coordinates.
        self.xyz_locations = np.array([[-1.,  0.,  0.],
                                        [ 1.,  0.,  0.],
                                        [ 0., -1.,  0.],
                                        [ 0.,  1.,  0.],
                                        [ 0.,  0., -1.],
                                        [ 0.,  0.,  1.]])

        self.llr_locations = np.array([[  0., 180., 1.],
                                        [  0.,   0., 1.],
                                        [  0., -90., 1.],
                                        [  0.,  90., 1.],
                                        [-90.,   0., 1.],
                                        [ 90.,   0., 1.]])

        self.xyz_locations_r = np.array([[-2.,  0., 0.],
                                        [ 2.,  0., 0.],
                                        [ 0., -2., 0.],
                                        [ 0.,  2., 0.],
                                        [ 0.,  0., -2.],
                                        [ 0.,  0., 2.]])

        self.llr_locations_r = np.array([[  0., 180., 2.],
                                        [  0.,   0., 2.],
                                        [  0., -90., 2.],
                                        [  0.,  90., 2.],
                                        [-90.,   0., 2.],
                                        [  90.,  0., 2.]])

        # cross distances between polar locations on a unit sphere in self.xyz_locations
        self.cross_distances = np.array([[0, 2, 1, 1, 1, 1],
                                        [2, 0, 1, 1, 1, 1],
                                        [1, 1, 0, 2, 1, 1],
                                        [1, 1, 2, 0, 1, 1],
                                        [1, 1, 1, 1, 0, 2],
                                        [1, 1, 1, 1, 2, 0]]) * np.pi/2

        # locations in arbitrary coordinates for tests involving pairs of locations
        self.locations_1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.locations_2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        self.locations_1_2D = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                                        [ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.locations_2_2D = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                                        [ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0]]).T

        # cross indices mapping between all location pairs in self.locations_1 and self.locations_2
        self.cross_indices_1 = np.array([[0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [2, 2, 2, 2, 2, 2],
                                        [3, 3, 3, 3, 3, 3],
                                        [4, 4, 4, 4, 4, 4],
                                        [5, 5, 5, 5, 5, 5],
                                        [6, 6, 6, 6, 6, 6]])

        self.cross_indices_2 = np.array([[0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5],
                                        [0, 1, 2, 3, 4, 5]])

        # vector pairs and angles between for vector_angle tests
        self.vectors_1 = np.array([[2.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0]])

        self.vectors_2 = np.array([[1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0]])

        self.vector_angles = np.array([0.0, np.pi, np.pi/2., np.pi/2., np.pi/2., np.pi/2.])


# and now call the class as a fixture
@pytest.fixture()
def example_data():
    return  Dummy()

def test_cartesian_to_polar2d(example_data: Dummy):
    """Test conversion of cartesian to 2D polar coordinates."""
    ll = geometry.cartesian_to_polar2d(example_data.xyz_locations)

    np.testing.assert_array_almost_equal(ll, example_data.llr_locations[:,:2])


def test_polar2d_to_cartesian(example_data: Dummy):
    """Test conversion of polar 2D to cartesian coordinates."""

    xyz = geometry.polar2d_to_cartesian(example_data.llr_locations[:,:2])
    np.testing.assert_array_almost_equal(xyz, example_data.xyz_locations)


def test_cross_indices(example_data: Dummy):
    """Test construction of indexing for all combination of two set of locations"""

    # locations in 1D form
    cross_1, cross_2 = geometry.cross_indices(example_data.locations_1, example_data.locations_2)

    np.testing.assert_array_equal(cross_1, example_data.cross_indices_1)
    np.testing.assert_array_equal(cross_2, example_data.cross_indices_2)

    # locations in 2D space should give the same cross indices
    cross_1_2D, cross_2_2D = geometry.cross_indices(example_data.locations_1_2D, example_data.locations_2_2D)

    np.testing.assert_array_equal(cross_1_2D, example_data.cross_indices_1)
    np.testing.assert_array_equal(cross_2_2D, example_data.cross_indices_2)


def test_vector_angle(example_data: Dummy):
    """Test angles between pairs of vectors"""

    vector_angles = geometry.vector_angle(example_data.vectors_1, example_data.vectors_2)
    np.testing.assert_array_almost_equal(vector_angles, example_data.vector_angles)


def test_cross_distance(example_data: Dummy):
    """Test computation of distance matrix between all pairs of locations in two location arrays"""

    cross_distances = geometry.cross_distance(example_data.xyz_locations)
    np.testing.assert_array_almost_equal(cross_distances, example_data.cross_distances)

    # test scale independence for output regardless of input vector scaling
    cross_distances = geometry.cross_distance(example_data.xyz_locations_r)
    np.testing.assert_array_almost_equal(cross_distances, example_data.cross_distances)

    # test scale dependence for output based of provided sphere radius
    cross_distances = geometry.cross_distance(example_data.xyz_locations_r, R=2.0)
    np.testing.assert_array_almost_equal(cross_distances, example_data.cross_distances*2.0)