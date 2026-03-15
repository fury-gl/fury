"""Geometry utilities for FURY."""

import numpy as np

from fury.lib import (
    Geometry,
)


def buffer_to_geometry(positions, **kwargs):
    """Convert a buffer to a geometry object.

    Parameters
    ----------
    positions : array_like
        The positions buffer.
    **kwargs : dict
        A dict of attributes to define on the geometry object. Keys can be
        "colors", "normals", "texcoords", "indices", etc.

    Returns
    -------
    Geometry
        The geometry object.

    Raises
    ------
    ValueError
        If positions array is empty or None.
    """
    if positions is None:
        raise ValueError("positions array cannot be empty or None.")

    geo = Geometry(positions=positions, **kwargs)
    return geo


def line_buffer_separator(line_vertices, color=None):
    """
    Create a line buffer with separators between segments.

    Parameters
    ----------
    line_vertices : list of array_like
        The line vertices as a list of segments (each segment is an array of points).
    color : array_like, optional
        The color of the line segments.

    Returns
    -------
    positions : array_like
        The positions buffer with NaN separators.
    colors : array_like, optional
        The colors buffer with NaN separators (if color is provided).
    """

    total_vertices = sum(len(segment) for segment in line_vertices)
    total_size = total_vertices + len(line_vertices) - 1

    positions_result = np.empty((total_size, 3), dtype=np.float32)

    if color is None:
        color = np.asarray((1, 1, 1, 1), dtype=np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)

    if (len(color) == 3 or len(color) == 4) and color.ndim == 1:
        color = np.tile(color, (len(line_vertices), 1))
        color_mode = "line"
    elif len(color) == len(line_vertices) and color.ndim == 2:
        color_mode = "line"
    elif len(color) == len(line_vertices) and color.ndim > 2:
        color_mode = "vertex"
    elif len(color) == total_vertices:
        color_mode = "vertex_flattened"
    else:
        raise ValueError(
            "Color array size doesn't match either vertex count or segment count"
        )

    colors_result = np.empty((total_size, color.shape[-1]), dtype=np.float32)

    idx = 0
    color_idx = 0

    for i, segment in enumerate(line_vertices):
        segment_length = len(segment)

        positions_result[idx : idx + segment_length] = segment

        if color_mode == "vertex":
            colors_result[idx : idx + segment_length] = color[i]
            color_idx += segment_length

        elif color_mode == "line":
            colors_result[idx : idx + segment_length] = np.tile(
                color[i], (segment_length, 1)
            )
        elif color_mode == "vertex_flattened":
            colors_result[idx : idx + segment_length] = color[
                color_idx : color_idx + segment_length
            ]
            color_idx += segment_length

        idx += segment_length

        if i < len(line_vertices) - 1:
            positions_result[idx] = np.nan
            colors_result[idx] = np.nan
            idx += 1

    return positions_result, colors_result


def rotate_vector(v, axis, angle):
    """Rotate a vector `v` around an axis `axis` by an angle `angle`.

    Parameters
    ----------
    v : array_like
        The vector to be rotated.
    axis : array_like
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    array_like
        The rotated vector.
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (
        v * cos_theta
        + np.cross(axis, v) * sin_theta
        + axis * np.dot(axis, v) * (1 - cos_theta)
    )


def prune_colinear(arr, colinear_threshold=0.9999):
    """Prune colinear points from the array.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        The input array of points.
    colinear_threshold : float, optional
        The threshold for colinearity. Points are considered colinear if the
        cosine of the angle between them is greater than or equal to this value.

    Returns
    -------
    ndarray, shape (3,)
        The pruned array with colinear points removed.
    """
    keep = [arr[0]]
    for i in range(1, len(arr) - 1):
        v1 = arr[i] - keep[-1]
        v2 = arr[i + 1] - arr[i]
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        if (
            np.linalg.norm(v1 / np.linalg.norm(v1) - v2 / np.linalg.norm(v2))
            >= colinear_threshold
        ):
            keep.append(arr[i])
    keep.append(arr[-1])
    return np.stack(keep)


def axes_for_dir(d, prev_x=None):
    """Compute the axes for a given direction vector.

    Parameters
    ----------
    d : ndarray, shape (3,)
        The direction vector.
    prev_x : ndarray, shape (3,), optional
        The previous x-axis vector.

    Returns
    -------
    x : ndarray, shape (3,)
        The x-axis vector.
    y : ndarray, shape (3,)
        The y-axis vector.
    """
    d /= np.linalg.norm(d)
    if prev_x is None:
        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(d, up)) > 0.9:
            up = np.array([0, 1, 0], dtype=np.float32)
        x = np.cross(up, d)
    else:
        x = prev_x - d * np.dot(prev_x, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return x, y


