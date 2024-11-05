"""Methods for geometric calculations on a sphere"""

# hadcrut/analysis/geometry.py @ 20200116

import numpy as np


def cross_indices(array_a: np.ndarray, array_b: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Construct an array of indices mapping between all combinations of elements of input arrays.
    Arg:
        array_a:
            A 2D array of shape (M, D) for arbitrary D.
        array_b:
            An optional second 1D array of shape (N, D) for arbitrary D. If provided then the returned then output
            index arrays will be of shape (M, N) mapping indices between array_a and array_b.
    Returns:
        A pair of arrays (indices_0, indices_1). If only array_a is provided then indices_0, indices_1 are each of
        dimension (M, M). If array_b is provided then indices_0, indices_1 are both of dimension (M, N).
    """

    a_len = array_a.shape[0]
    if array_b is None:
        b_len = a_len
    else:
        b_len = array_b.shape[0]

    indices_0, indices_1 = np.mgrid[0:a_len, 0:b_len]

    return indices_0, indices_1


def cartesian_to_polar2d(V: np.ndarray) -> np.ndarray:
    """Convert 3D cartesian locations in V to polar coordinates.
    Uses the convention that x-axis is 0E -> 0N, y-axis is 90E -> 0N, and z is 0E -> 90N.
    Args:
        V (np.ndarray):
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z].
    Returns:
        Array of dimension (M, 2) containing (latitude, longitude) pairs for each location in V. Expressed in degrees.
    """

    radius = np.linalg.norm(V, axis=1)
    latitude = np.arcsin(V[:, 2] / radius)
    longitude = np.arctan2(V[:, 1], V[:, 0])

    return np.degrees(np.vstack((latitude, longitude)).T)


def polar2d_to_cartesian(polar2d: np.ndarray) -> np.ndarray:
    """Convert 2D polar coordinates to cartesian coordinates on a unit sphere.
    Args:
        polar2d:
            Array of dimension (M, 2) containing M (latitude, longitude) pairs. Expressed in degrees.
    Returns:
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z] on the
            surface of a unit sphere corresponding to the latitude, longitude pairs in polar2d.
    """

    z = np.sin(np.radians(polar2d[:, 0]))
    minor_r = np.cos(np.radians(polar2d[:, 0]))
    x = np.cos(np.radians(polar2d[:, 1])) * minor_r
    y = np.sin(np.radians(polar2d[:, 1])) * minor_r

    return np.vstack((x, y, z)).T


def cartesian_to_polar3d(V: np.ndarray) -> np.ndarray:
    """Convert 3D cartesian locations in V to polar coordinates.
    Using the convention describing vectors from the origin as (latitude, longitude, length), unit vectors along each
    axis define the axes as:
        x-axis: vector from the origin to (0, 0, 1)
        y-axis: vector from the origin to (0, 90, 1)
        z-axis: vector from the origin to (90, 0, 1)
    Args:
        V (np.ndarray):
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z].
    Returns:
        Array of dimension (M, 3) containing (latitude, longitude, radius) pairs for each location in V. Expressed in
        degrees.
    """

    radius = np.linalg.norm(V, axis=1)
    latitude = np.arcsin(V[:, 2] / radius)
    longitude = np.arctan2(V[:, 1], V[:, 0])

    return np.vstack((np.degrees(latitude), np.degrees(longitude), radius)).T


def polar3d_to_cartesian(polar3d: np.ndarray) -> np.ndarray:
    """Convert 3D polar coordinates to cartesian coordinates on a unit sphere.
    Args:
        polar3d:
            Array of dimension (M, 3) containing M (latitude, longitude, radius) triplets. Angles expressed in degrees.
    Returns:
            Array of dimension (M, 3) representing M locations in 3D space with spatial dimensions [x, y, z] on the
            surface of a unit sphere corresponding to the latitude, longitude pairs in polar2d.
    """

    major_r = polar3d[:, 2]
    z = np.sin(np.radians(polar3d[:, 0])) * major_r
    minor_r = np.cos(np.radians(polar3d[:, 0])) * major_r
    x = np.cos(np.radians(polar3d[:, 1])) * minor_r
    y = np.sin(np.radians(polar3d[:, 1])) * minor_r

    return np.vstack((x, y, z)).T


def vector_angle(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute angles between N pairs of vectors in arrays X and Y.
    Uses arctan2 form that is numerically stable for small angles.
    Args:
        X:
            A numpy.array of dimension (N, D) where N is the number of vectors and D is
            the dimension of the space, e.g. 3 for locations in 3D.
        Y:
            A numpy.array of dimension (N, D) where N is the number of vectors and D is
            the dimension of the space, e.g. 3 for locations in 3D.
    Returns:
        A numpy array of length n containing angles between X and Y in radians. Output in radians.
    """

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)

    return 2 * np.arctan2(np.linalg.norm((X.T * Y_norm).T - (Y.T * X_norm).T, axis=1),
                          np.linalg.norm((X.T * Y_norm).T + (Y.T * X_norm).T, axis=1))


def cross_distance(locations_a: np.ndarray, locations_b: np.ndarray = None, R: int = 1.0) -> np.ndarray:
    """Compute cross distances on a sphere between all combinations of input locations.
    Args:
        locations_a:
            Array of locations in 3D space of shape (M,3). If provided then distances are returned for all
            combinations of locations in (locations_a, locations_a). Cartesian coordinates.
    Kwargs:
        locations_b:
            Optional second array of locations of shape (N,3).  If provided then distances are returned for all
            combinations of locations in (locations_a, locations_b). Cartesian coordinates.
        R:
            Radius of the sphere. Defaults to R=1.0.
    Returns:
        A cross distance matrix containing all combinations of distances between locations in locations_a and itself as
        an (M, M) array. If locations_b is provided then an array of shape (M, N) is returned containing all
        if provided the all possible pairs of locations in (locations_a, locations_b).
    """

    if locations_b is None:
        locations_b = locations_a

    indices_a, indices_b = cross_indices(locations_a, locations_b)

    angle_matrix = vector_angle(locations_a[indices_a.ravel(), :],
                                locations_b[indices_b.ravel(), :]).reshape(locations_a.shape[0], locations_b.shape[0])

    return R * angle_matrix
