"""
Billboard actor module.

Minimal isolated implementation of billboard support to reduce diffs in
existing planar actor module. Provides a Mesh-based world object and a
factory function plus shader registration.
"""

import numpy as np

from fury.actor import Mesh
from fury.geometry import buffer_to_geometry
from fury.lib import register_wgpu_render_function
from fury.material import (
    BillboardMaterial,
    BillboardSphereMaterial,
    validate_opacity,
)
from fury.shader import BillboardShader, BillboardSphereShader


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
    """
    Build a ``Billboard`` instance from broadcasted inputs.

    Parameters
    ----------
    centers : array_like
        Position of each billboard specified as an ``(N, 3)`` array or
        broadcastable equivalent.
    colors : array_like
        RGB or RGBA color per billboard. A single color is broadcast when
        needed.
    sizes : array_like
        Width and height per billboard. Accepts scalar, ``(2,)`` pair,
        ``(N,)`` radius (interpreted as square billboards), or ``(N, 2)`` data.
    opacity : float or None
        Global opacity multiplier. ``None`` keeps the material default.
    enable_picking : bool
        Whether the billboard should write picking information.
    material_cls : type[BillboardMaterial]
        Material class used to instantiate the billboard actor.
    material_kwargs : dict, optional
        Additional keyword arguments forwarded to ``material_cls``.

    Returns
    -------
    Billboard
        Configured billboard world object containing geometry, material and
        metadata about the generated billboards.
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
    """
    World object representing one or more billboards.

    Geometry buffers are duplicated per 6 vertices (two triangles) per
    billboard; the vertex shader reconstructs the quad via ``vertex_index``
    math and uses camera right/up vectors to orient it. Size metadata is
    stored on ``billboard_sizes`` and reused by shaders for impostor variants.

    Parameters
    ----------
    geometry : Geometry, optional
        The geometry object containing vertex data.
    material : Material, optional
        The material used to render the billboard.
    **kwargs : dict
        Additional keyword arguments forwarded to :class:`Mesh`.
    """

    def __init__(self, geometry=None, material=None, **kwargs):
        """Initialize a Billboard actor."""
        super().__init__(geometry=geometry, material=material, **kwargs)
        self._billboard_count = 0
        self._billboard_centers = np.empty((0, 3), dtype=np.float32)
        self._billboard_sizes = np.empty((0, 2), dtype=np.float32)

    def get_bounding_box(self):
        """
        Compute the axis-aligned bounding box including billboard visual size.

        Returns
        -------
        ndarray, shape (2, 3), or None
            Axis-aligned bounding box as ``[[min, min, min], [max, max, max]]``.
        """
        centers = self._billboard_centers
        sizes = self._billboard_sizes

        if centers.size == 0 or sizes.size == 0:
            return super().get_bounding_box()

        half_extents = sizes.max(axis=1) / 2.0

        expanded_min = centers - half_extents[:, np.newaxis]
        expanded_max = centers + half_extents[:, np.newaxis]

        aabb = np.empty((2, 3), dtype=np.float64)
        aabb[0] = expanded_min.min(axis=0)
        aabb[1] = expanded_max.max(axis=0)

        parent_aabb = super().get_bounding_box()
        if parent_aabb is not None:
            aabb[0] = np.minimum(aabb[0], parent_aabb[0])
            aabb[1] = np.maximum(aabb[1], parent_aabb[1])

        return aabb

    @property
    def billboard_count(self):
        """
        Get the number of billboards in this actor.

        Returns
        -------
        int
            Number of billboards.
        """
        return self._billboard_count

    @billboard_count.setter
    def billboard_count(self, value):
        """
        Set the number of billboards in this actor.

        Parameters
        ----------
        value : int
            Number of billboards.
        """
        self._billboard_count = value

    @property
    def billboard_centers(self):
        """
        Get the billboard center positions.

        Returns
        -------
        ndarray, shape (N, 3)
            Billboard center positions.
        """
        return self._billboard_centers

    @billboard_centers.setter
    def billboard_centers(self, value):
        """
        Set the billboard center positions.

        Parameters
        ----------
        value : array_like, shape (N, 3)
            Billboard center positions.
        """
        self._billboard_centers = np.asarray(value, dtype=np.float32)

    @property
    def billboard_sizes(self):
        """
        Get the billboard width/height pairs.

        Returns
        -------
        ndarray, shape (N, 2)
            Billboard width/height pairs.
        """
        return self._billboard_sizes

    @billboard_sizes.setter
    def billboard_sizes(self, value):
        """
        Set the billboard width/height pairs.

        Parameters
        ----------
        value : array_like, shape (N, 2)
            Billboard width/height pairs.
        """
        self._billboard_sizes = np.asarray(value, dtype=np.float32)


def billboard(
    centers,
    *,
    colors=(1, 1, 1),
    sizes=(1, 1),
    opacity=None,
    enable_picking=True,
):
    """
    Create a billboard world object.

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


def billboard_sphere(
    centers,
    *,
    colors=(1, 1, 1),
    radii=0.5,
    opacity=None,
    enable_picking=True,
):
    """
    Create a billboard impostor sphere world object.

    Parameters
    ----------
    centers : array_like
        Sphere centers provided as an ``(N, 3)`` array or broadcastable input.
    colors : array_like, optional
        RGB or RGBA color per sphere. Single color inputs are broadcast.
    radii : array_like, optional
        Scalar radii or per-sphere radii array. Used to compute billboard size.
    opacity : float, optional
        Opacity multiplier applied to the material.
    enable_picking : bool, optional
        Whether the impostor spheres support picking.

    Returns
    -------
    Billboard
        Billboard actor configured to simulate spheres using impostor quads.
    """

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
    """
    Build the render pipeline for ``Billboard`` instances.

    Parameters
    ----------
    wobject : Billboard
        Billboard world object to bind to the shader pipeline.

    Returns
    -------
    tuple
        Tuple containing the configured shader instance.
    """
    return (BillboardShader(wobject),)


@register_wgpu_render_function(Billboard, BillboardSphereMaterial)
def register_billboard_sphere_render_function(wobject):
    """
    Register the pipeline for billboard-based sphere impostors.

    Parameters
    ----------
    wobject : Billboard
        Billboard world object representing impostor spheres.

    Returns
    -------
    tuple
        Tuple containing the configured
        :class:`~fury.shader.BillboardSphereShader`.
    """
    return (BillboardSphereShader(wobject),)
