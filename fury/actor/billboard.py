"""Billboard actor module.

Minimal isolated implementation of billboard support to reduce diffs in
existing planar actor module. Provides a Mesh-based world object and a
factory function plus shader registration.
"""

from __future__ import annotations

import numpy as np

from fury.geometry import buffer_to_geometry
from fury.lib import Mesh, register_wgpu_render_function
from fury.material import BillboardMaterial, validate_opacity


class Billboard(Mesh):
    """World object representing one or more billboards.

    Geometry buffers are duplicated per 6 vertices (two triangles) per
    billboard; the vertex shader reconstructs the quad via ``vertex_index``
    math and uses camera right/up vectors to orient it.
    """

    pass


def billboard(
    centers,
    *,
    colors=(1, 1, 1),
    sizes=(1, 1),
    opacity=None,
    enable_picking=True,
):
    """Create a billboard world object.

    Parameters
    ----------
    centers : (N,3) array_like
        Billboard positions.
    colors : (N,3|4) array_like or single color
        Per-billboard RGB(A) colors.
    sizes : (N,2) | (2,) | float | (N,) array_like
        Width/height per billboard. Scalar or single pair broadcast.
    opacity : float, optional
        Global opacity multiplier (0..1).
    enable_picking : bool
        Whether billboard is pickable.

    Returns
    -------
    Billboard
        Billboard world object configured with the provided geometry and
        material.
    """
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, 3)
    n = len(centers)

    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1:
        colors = np.tile(colors, (n, 1))
    elif colors.shape[0] != n:
        colors = np.tile(colors[0], (n, 1))

    sizes = np.asarray(sizes, dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full((n, 2), float(sizes))
    elif sizes.ndim == 1:
        if sizes.size == 2:
            sizes = np.tile(sizes, (n, 1))
        elif sizes.size == n:  # per-billboard square
            sizes = np.column_stack([sizes, sizes])
        else:
            sizes = np.full((n, 2), sizes.flat[0])
    elif sizes.shape[0] != n:
        sizes = np.tile(sizes[0], (n, 1))

    opacity = validate_opacity(opacity)

    repeats = 6  # 2 triangles per quad
    pos = np.repeat(centers, repeats, axis=0).astype(np.float32)
    col = np.repeat(colors, repeats, axis=0).astype(np.float32)
    sz = np.repeat(sizes, repeats, axis=0).astype(np.float32)
    indices = np.arange(pos.shape[0], dtype=np.uint32)

    normals = np.zeros((pos.shape[0], 3), dtype=np.float32)
    normals[:, 0:2] = sz

    geometry = buffer_to_geometry(
        positions=pos,
        colors=col,
        normals=normals,
        indices=indices,
    )

    material = BillboardMaterial(
        pick_write=enable_picking,
        opacity=opacity,
        color_mode="vertex",
    )

    obj = Billboard(geometry=geometry, material=material)
    obj.billboard_count = n
    obj.billboard_centers = centers.copy()
    return obj


@register_wgpu_render_function(Billboard, BillboardMaterial)
def register_billboard_render_function(wobject):
    """Build the render pipeline for ``Billboard`` instances.

    Parameters
    ----------
    wobject : Billboard
        Billboard world object to bind to the shader pipeline.

    Returns
    -------
    tuple
        Tuple containing the configured shader instance.
    """
    from fury.shader import BillboardShader

    return (BillboardShader(wobject),)
