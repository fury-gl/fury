"""Billboard actor module.

Minimal isolated implementation of billboard support to reduce diffs in
existing planar actor module. Provides a Mesh-based world object and a
factory function plus shader registration.
"""

from __future__ import annotations

import numpy as np

from fury.actor import Mesh
from fury.geometry import buffer_to_geometry
from fury.lib import register_wgpu_render_function
from fury.material import (
    BillboardMaterial,
    BillboardSphereMaterial,
    validate_opacity,
)


def _create_billboard_actor(
    centers,
    colors,
    sizes,
    opacity,
    enable_picking,
    *,
    material_cls,
    material_kwargs=None,
):
    """Build a ``Billboard`` instance from broadcasted inputs.

    The helper normalizes the array inputs, generates the shared geometry and
    instantiates the requested material. It keeps the original function small
    and reusable for alternative billboard-based actors (e.g. spheres).
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
        elif sizes.size == n:
            sizes = np.column_stack([sizes, sizes])
        else:
            sizes = np.full((n, 2), sizes.flat[0])
    elif sizes.shape[0] != n:
        sizes = np.tile(sizes[0], (n, 1))

    opacity = validate_opacity(opacity)

    repeats = 6  # 2 triangles per quad
    pos = np.repeat(centers, repeats, axis=0).astype(np.float32)
    col = np.repeat(colors, repeats, axis=0).astype(np.float32)
    indices = np.arange(pos.shape[0], dtype=np.uint32)

    # Encode per-billboard size in normals so shaders can fetch dimensions
    normals = np.repeat(
        np.column_stack([sizes, np.ones((n, 1), dtype=np.float32)]),
        repeats,
        axis=0,
    ).astype(np.float32)

    geometry = buffer_to_geometry(
        positions=pos,
        colors=col,
        normals=normals,
        indices=indices,
    )

    material_kwargs = material_kwargs or {}
    material = material_cls(
        pick_write=enable_picking,
        opacity=opacity,
        color_mode="vertex",
        **material_kwargs,
    )

    obj = Billboard(geometry=geometry, material=material)
    obj.billboard_count = n
    obj.billboard_centers = centers.copy()
    obj.billboard_sizes = sizes.copy()
    return obj


class Billboard(Mesh):
    """World object representing one or more billboards.

    Geometry buffers are duplicated per 6 vertices (two triangles) per
    billboard; the vertex shader reconstructs the quad via ``vertex_index``
    math and uses camera right/up vectors to orient it. Size metadata is
    stored on ``billboard_sizes`` and reused by shaders for impostor variants.
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
    return _create_billboard_actor(
        centers,
        colors,
        sizes,
        opacity,
        enable_picking,
        material_cls=BillboardMaterial,
    )


def create_billboard_sphere(
    centers,
    *,
    colors=(1, 1, 1),
    radii=0.5,
    opacity=None,
    enable_picking=True,
):
    """Create a billboard impostor sphere world object."""

    sizes = np.asarray(radii, dtype=np.float32) * 2.0
    obj = _create_billboard_actor(
        centers,
        colors,
        sizes,
        opacity,
        enable_picking,
        material_cls=BillboardSphereMaterial,
    )
    obj.billboard_radii = obj.billboard_sizes[:, 0] * 0.5
    obj.billboard_mode = "impostor"
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


@register_wgpu_render_function(Billboard, BillboardSphereMaterial)
def register_billboard_sphere_render_function(wobject):
    from fury.shader import BillboardSphereShader

    return (BillboardSphereShader(wobject),)
