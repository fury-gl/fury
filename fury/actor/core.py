# -*- coding: utf-8 -*-
"""Core actor functionality for FURY."""

import numpy as np

from fury.geometry import (
    buffer_to_geometry,
    create_line,
    create_mesh,
    line_buffer_separator,
)
from fury.material import _create_line_material, _create_mesh_material
import fury.primitive as fp


def actor_from_primitive(
    vertices,
    faces,
    centers,
    *,
    colors=(1, 0, 0),
    scales=(1, 1, 1),
    directions=(1, 0, 0),
    opacity=None,
    material="phong",
    smooth=False,
    enable_picking=True,
    repeat_primitive=True,
):
    """Build an actor from a primitive.

    Parameters
    ----------
    vertices : ndarray
        Vertices of the primitive.
    faces : ndarray
        Faces of the primitive.
    centers : ndarray, shape (N, 3)
        Primitive positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the primitive in each dimension. If a single value is provided,
        the same size will be used for all primitives.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the primitive.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the primitive. Options are 'phong' and 'basic'.
    smooth : bool, optional
        Whether to create a smooth primitive or a faceted primitive.
    enable_picking : bool, optional
        Whether the primitive should be pickable in a 3D scene.
    repeat_primitive : bool, optional
        Whether to repeat the primitive for each center. If False,
        only one instance of the primitive is created at the first center.

    Returns
    -------
    Actor
        A mesh actor containing the generated primitive, with the specified
        material and properties.
    """

    if repeat_primitive:
        res = fp.repeat_primitive(
            vertices,
            faces,
            centers,
            directions=directions,
            colors=colors,
            scales=scales,
        )
        big_vertices, big_faces, big_colors, _ = res

    else:
        big_vertices = vertices
        big_faces = faces
        big_colors = colors

    prim_count = len(centers)

    if isinstance(opacity, (int, float)):
        if big_colors.shape[1] == 3:
            big_colors = np.hstack(
                (big_colors, np.full((big_colors.shape[0], 1), opacity))
            )
        else:
            big_colors[:, 3] *= opacity

    geo = buffer_to_geometry(
        indices=big_faces.astype("int32"),
        positions=big_vertices.astype("float32"),
        texcoords=big_vertices.astype("float32"),
        colors=big_colors.astype("float32"),
    )

    mat = _create_mesh_material(
        material=material, enable_picking=enable_picking, flat_shading=not smooth
    )
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count
    return obj


def arrow(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    height=1.0,
    resolution=10,
    tip_length=0.35,
    tip_radius=0.1,
    shaft_radius=0.03,
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Create one or many arrows with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Arrow positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the arrow.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    height : float, optional
        The total height of the arrow, including the shaft and tip.
    resolution : int, optional
        The number of divisions along the arrow's circular cross-sections.
        Higher values produce smoother arrows.
    tip_length : float, optional
        The length of the arrowhead tip relative to the total height.
    tip_radius : float, optional
        The radius of the arrowhead tip.
    shaft_radius : float, optional
        The radius of the arrow shaft.
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the arrow in each dimension. If a single value is
        provided, the same size will be used for all arrows.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the arrows. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the arrows should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated arrows, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> arrow_actor = actor.arrow(centers=centers, colors=colors)
    >>> _ = scene.add(arrow_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    vertices, faces = fp.prim_arrow(
        height=height,
        resolution=resolution,
        tip_length=tip_length,
        tip_radius=tip_radius,
        shaft_radius=shaft_radius,
    )
    return actor_from_primitive(
        vertices,
        faces,
        centers=centers,
        colors=colors,
        scales=scales,
        directions=directions,
        opacity=opacity,
        material=material,
        enable_picking=enable_picking,
    )


def axes(
    *,
    scale=(1.0, 1.0, 1.0),
    color_x=(1.0, 0.0, 0.0),
    color_y=(0.0, 1.0, 0.0),
    color_z=(0.0, 0.0, 1.0),
    opacity=1.0,
):
    """Create coordinate system axes using colored arrows.

    The axes are represented as arrows with different colors:
    red = X-axis, green = Y-axis, blue = Z-axis.

    Parameters
    ----------
    scale : tuple (3,), optional
        The size (length) of each axis in the x, y, and z directions.
    color_x : tuple (3,), optional
        Color for the X-axis.
    color_y : tuple (3,), optional
        Color for the Y-axis.
    color_z : tuple (3,), optional
        Color for the Z-axis.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    Actor
        An axes actor representing the coordinate axes with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> axes_actor = actor.axes()
    >>> _ = scene.add(axes_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    centers = np.zeros((3, 3))
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array(
        [color_x + (opacity,), color_y + (opacity,), color_z + (opacity,)]
    )
    scales = np.asarray(scale)

    obj = arrow(centers=centers, directions=directions, colors=colors, scales=scales)
    return obj


def line(
    lines,
    *,
    colors=(1, 0, 0),
    opacity=None,
    material="basic",
    enable_picking=True,
):
    """
    Visualize one or many lines with different colors.

    Parameters
    ----------
    lines : list of ndarray of shape (P, 3) or ndarray of shape (N, P, 3)
        Lines points.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    material : str, optional
        The material type for the lines. Options are 'basic', 'segment', 'arrow',
        'thin', and 'thin_segment'.
    enable_picking : bool, optional
        Whether the lines should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated lines, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> lines = [np.random.rand(10, 3) for _ in range(5)]
    >>> colors = np.random.rand(5, 3)
    >>> line_actor = actor.line(lines=lines, colors=colors)
    >>> _ = scene.add(line_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    lines_positions, lines_colors = line_buffer_separator(
        lines, color=colors, color_mode="auto"
    )

    geo = buffer_to_geometry(
        positions=lines_positions.astype("float32"),
        colors=lines_colors.astype("float32")
        if lines_colors is not None
        else np.empty_like(lines_positions),
    )

    if lines_colors is None:
        material_mode = "auto"
        material_colors = None
    else:
        material_mode = "vertex"
        material_colors = lines_colors

    mat = _create_line_material(
        material=material,
        enable_picking=enable_picking,
        mode=material_mode,
        opacity=opacity,
        color=material_colors,
    )

    obj = create_line(geometry=geo, material=mat)

    obj.local.position = lines_positions[0]

    obj.prim_count = len(lines)

    return obj
