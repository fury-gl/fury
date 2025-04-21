"""Geometry utilities for FURY."""

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation

from fury.lib import (
    Geometry,
    Line,
    Mesh,
    MeshBasicMaterial,
    MeshPhongMaterial,
    Points,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    Text,
    TextMaterial,
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
    if positions is None or positions.size == 0:
        raise ValueError("positions array cannot be empty or None.")

    geo = Geometry(positions=positions, **kwargs)
    return geo


def create_mesh(geometry, material):
    """Create a mesh object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be either MeshPhongMaterial or MeshBasicMaterial.

    Returns
    -------
    Mesh
        The mesh object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of MeshPhongMaterial or MeshBasicMaterial.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(material, (MeshPhongMaterial, MeshBasicMaterial)):
        raise TypeError(
            "material must be an instance of MeshPhongMaterial or MeshBasicMaterial."
        )

    mesh = Mesh(geometry=geometry, material=material)
    return mesh


def create_line(geometry, material):
    """
    Create a line object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object.

    Returns
    -------
    Line
        The line object.
    """
    line = Line(geometry=geometry, material=material)
    return line


def line_buffer_separator(line_vertices, color=None, color_mode="auto"):
    """
    Create a line buffer with separators between segments.

    Parameters
    ----------
    line_vertices : list of array_like
        The line vertices as a list of segments (each segment is an array of points).
    color : array_like, optional
        The color of the line segments.
    color_mode : str, optional
        The color mode, can be 'auto', 'vertex', or 'line'.
        - 'auto': Automatically determine based on color array shape
        - 'vertex': One color per vertex (must match total vertex count)
        - 'line': One color per line segment

    Returns
    -------
    positions : array_like
        The positions buffer with NaN separators.
    colors : array_like, optional
        The colors buffer with NaN separators (if color is provided).
    """
    # Calculate total size including separators
    total_vertices = sum(len(segment) for segment in line_vertices)
    total_size = total_vertices + len(line_vertices) - 1

    positions_result = np.empty((total_size, 3), dtype=np.float32)
    colors_result = None

    if color is not None:
        colors_result = np.empty((total_size, 3), dtype=np.float32)
        if color_mode == "auto":
            if len(color) == len(line_vertices) and (
                len(color[0]) == 3 or len(color[0]) == 4
            ):
                color_mode = "line"
            elif len(color) == total_vertices:
                color_mode = "vertex_flattened"
            elif len(color) == len(line_vertices):
                color_mode = "vertex"
            elif len(color) == 3 or len(color) == 4:
                color = None
            else:
                raise ValueError(
                    "Color array size doesn't match "
                    "either vertex count or segment count"
                )

    idx = 0
    color_idx = 0

    for i, segment in enumerate(line_vertices):
        segment_length = len(segment)

        positions_result[idx : idx + segment_length] = segment

        if color is not None:
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
            else:
                raise ValueError("Invalid color mode")

        idx += segment_length

        if i < len(line_vertices) - 1:
            positions_result[idx] = np.nan
            if color is not None:
                colors_result[idx] = np.nan
            idx += 1

    return positions_result, colors_result if color is not None else None


def create_point(geometry, material):
    """Create a point object.

    Parameters
    ----------
    geometry : Geometry
        The geometry object.
    material : Material
        The material object. Must be either PointsMaterial, PointsGaussianBlobMaterial,
        or PointsMarkerMaterial.

    Returns
    -------
    Points
        The point object.

    Raises
    ------
    TypeError
        If geometry is not an instance of Geometry or material is not an
        instance of PointsMaterial, PointsGaussianBlobMaterial, or PointsMarkerMaterial.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be an instance of Geometry.")

    if not isinstance(
        material, (PointsMaterial, PointsGaussianBlobMaterial, PointsMarkerMaterial)
    ):
        raise TypeError(
            "material must be an instance of PointsMaterial, "
            "PointsGaussianBlobMaterial or PointsMarkerMaterial."
        )

    point = Points(geometry=geometry, material=material)
    return point


def create_text(text, material, **kwargs):
    """Create a text object.

    Parameters
    ----------
    text : str
        The text content.
    material : TextMaterial
        The material object.
    **kwargs : dict
        Additional properties like font_size, anchor, etc.

    Returns
    -------
    Text
        The text object.

    Raises
    ------
    TypeError
        If text is not a string or material is not an instance of TextMaterial.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(material, TextMaterial):
        raise TypeError("material must be an instance of TextMaterial.")

    text = Text(text=text, material=material, **kwargs)
    return text


def _tube_frame(frames, tangents):
    """
    Calculates a frame along a curve defined by tangents.

    Parameters
    ----------
    frames : list
        An empty list to be filled with (n, b) tuples representing the normal
        and bi normal vectors at each point.
    tangents : ndarray (N, 3)
        The tangent vectors along the curve.
    """
    t0 = tangents[0] / np.linalg.norm(tangents[0])
    if np.linalg.norm(t0) < 1e-6:
        cross_vec = (
            np.array([0, 0, 1])
            if np.linalg.norm(np.cross(t0, np.array([0, 1, 0]))) < 1e-6
            else np.array([0, 1, 0])
        )
        n0 = np.cross(t0, cross_vec)
    else:
        n0 = np.cross(t0, np.array([0, 1, 0]))
        if np.linalg.norm(n0) < 1e-6:
            n0 = np.cross(t0, np.array([0, 0, 1]))

    n0 = n0 / np.linalg.norm(n0)
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.array([1.0, 0.0, 0.0])

    b0 = np.cross(t0, n0)
    b0 = b0 / np.linalg.norm(b0)
    if np.linalg.norm(b0) < 1e-6:
        b0 = np.array([0.0, 1.0, 0.0])

    frames.append((n0, b0))

    for i in range(1, len(tangents)):
        t1 = tangents[i] / np.linalg.norm(tangents[i])
        if np.linalg.norm(t1) < 1e-6:
            frames.append(frames[-1])
            t0 = t1
            continue

        axis = np.cross(t0, t1)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-6:
            frames.append(frames[-1])
            t0 = t1
            continue

        axis /= axis_len
        angle = np.arctan2(axis_len, np.dot(t0, t1))
        rot = Rotation.from_rotvec(axis * angle)

        # n_rot = rot.apply(frames[-1][0])
        b_rot = rot.apply(frames[-1][1])

        t_new = t1
        n_new = np.cross(b_rot, t_new)
        n_new = n_new / np.linalg.norm(n_new)
        if np.linalg.norm(n_new) < 1e-6:
            n_new = frames[-1][0]

        b_new = np.cross(t_new, n_new)
        b_new = b_new / np.linalg.norm(b_new)
        if np.linalg.norm(b_new) < 1e-6:
            b_new = frames[-1][1]

        frames.append((n_new, b_new))
        t0 = t_new


def _generate_smooth_points(points, num_interpolation_points=10):
    """
    Generates smooth points along a curve using spline interpolation.

    Parameters
    ----------
    points : ndarray (N, 3)
        The control points for the spline.
    num_interpolation_points : int, optional
        The number of interpolation points to generate between each pair
        of original points.

    Returns
    -------
    smooth_points : ndarray (M, 3)
        The array of smoothly interpolated points.
    """
    if len(points) < 2:
        return np.array(points, dtype=np.float32)

    tck, u = splprep(points.T, s=0)

    u_new = np.linspace(u.min(), u.max(), len(points) * num_interpolation_points)
    smooth_points = np.array(splev(u_new, tck)).T

    return smooth_points
