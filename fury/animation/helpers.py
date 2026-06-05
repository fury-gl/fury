"""Interpolation functions for keyframes."""

import numpy as np
from scipy.spatial import transform


def compose_transform_matrix(
    position=None, rotation_quat=None, scale_factors=None, parent_matrix=None
):
    """
    Compose a 4x4 transformation matrix from position, rotation, and scale.

    Parameters
    ----------
    position : ndarray, shape (3,), optional
        Translation vector.
    rotation_quat : ndarray, shape (4,), optional
        Rotation quaternion [x, y, z, w].
    scale_factors : ndarray, shape (3,), optional
        Scale factors for x, y, z.
    parent_matrix : ndarray, shape (4, 4), optional
        Parent transformation matrix to multiply with.

    Returns
    -------
    ndarray, shape (4, 4)
        The composed transformation matrix.
    """
    matrix = np.identity(4, dtype=np.float32)

    if parent_matrix is not None:
        matrix = parent_matrix.copy()

    if scale_factors is not None:
        scale_mat = np.identity(4, dtype=np.float32)
        scale_mat[0, 0] = scale_factors[0]
        scale_mat[1, 1] = scale_factors[1]
        scale_mat[2, 2] = scale_factors[2]
        matrix = matrix @ scale_mat

    if rotation_quat is not None:
        rot = transform.Rotation.from_quat(rotation_quat)
        rot_mat = np.identity(4, dtype=np.float32)
        rot_mat[:3, :3] = rot.as_matrix()
        matrix = matrix @ rot_mat

    if position is not None:
        trans_mat = np.identity(4, dtype=np.float32)
        trans_mat[:3, 3] = position
        matrix = trans_mat @ matrix

    return matrix


def get_previous_timestamp(timestamps, current_time, *, include_last=False):
    """
    Return the maximum previous timestamp of a given time.

    Parameters
    ----------
    timestamps : ndarray
        Sorted list of timestamps.
    current_time : float or int
        The time to get previous timestamp for.
    include_last : bool, optional
        If `True`, even the last timestamp will be considered a valid previous
        timestamp.

    Returns
    -------
    float or int
        The previous timestamp.
    """
    for timestamp in timestamps[::-1] if include_last else timestamps[-2::-1]:
        if timestamp <= current_time:
            return timestamp
    return timestamps[0]


def get_next_timestamp(timestamps, current_time, *, include_first=False):
    """
    Return the minimum next timestamp of a given time.

    Parameters
    ----------
    timestamps : ndarray
        Sorted list of timestamps.
    current_time : float or int
        The time to get previous timestamp for.
    include_first : bool, optional
        If `True`, even the first timestamp will be considered a valid next
        timestamp.

    Returns
    -------
    float or int
        The next timestamp.
    """
    for timestamp in timestamps[:] if include_first else timestamps[1:]:
        if timestamp > current_time:
            return timestamp
    return timestamps[-1]


def get_timestamps_from_keyframes(keyframes):
    """
    Return a sorted array of timestamps given dict of keyframes.

    Parameters
    ----------
    keyframes : dict
        Keyframes dict that contains timestamps as keys.

    Returns
    -------
    ndarray
        Array of sorted timestamps extracted from the keyframes.
    """
    return np.sort(np.array(list(keyframes)), axis=None)


def get_values_from_keyframes(keyframes):
    """
    Return an array of keyframes values sorted using timestamps.

    Parameters
    ----------
    keyframes : dict
        Keyframes dict that contains timestamps as keys and data as values.

    Returns
    -------
    ndarray
        Array of sorted values extracted from the keyframes.
    """
    return np.asarray(
        [keyframes.get(t, {}).get("value", None) for t in sorted(keyframes.keys())]
    )


def get_time_tau(t, t0, t1):
    """
    Return a capped time tau between 0 and 1.

    Parameters
    ----------
    t : float or int
        Current time to calculate tau for.
    t0 : float or int
        Lower timestamp of the time period.
    t1 : float or int
        Higher timestamp of the time period.

    Returns
    -------
    float
        The time tau.
    """
    return 0 if t <= t0 else 1 if t >= t1 else (t - t0) / (t1 - t0)


def lerp(v0, v1, t0, t1, t):
    """
    Return a linearly interpolated value.

    Parameters
    ----------
    v0 : ndarray or float or int
        The first value.
    v1 : ndarray or float or int
        The second value.
    t0 : float or int
        Timestamp associated with v0.
    t1 : float or int
        Timestamp associated with v1.
    t : float or int
        Current time to interpolate at.

    Returns
    -------
    ndarray or float
        The interpolated value.
    """
    if t0 == t1:
        return v0
    v = v1 - v0
    dt = get_time_tau(t, t0, t1)
    return dt * v + v0


def euclidean_distances(points):
    """
    Return a list of euclidean distances of a list of points or values.

    Parameters
    ----------
    points : ndarray
        Array of points or valued to calculate euclidean distances between.

    Returns
    -------
    list
        A List of euclidean distance between each consecutive points or values.
    """
    return [np.linalg.norm(x - y) for x, y in zip(points, points[1:], strict=False)]
