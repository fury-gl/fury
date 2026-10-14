"""
Billboard actor module.

Minimal isolated implementation of billboard support to reduce diffs in
existing planar actor module. Provides a Mesh-based world object and a
factory function plus shader registration.
"""

import numpy as np

from fury.actor import Mesh
from fury.colormap import normalize_colors
from fury.geometry import buffer_to_geometry
from fury.lib import gfx_wgpu, register_wgpu_render_function
from fury.material import (
    BillboardEllipsoidMaterial,
    BillboardMaterial,
    BillboardSphereMaterial,
    validate_opacity,
)
from fury.shader import (
    BillboardEllipsoidShader,
    BillboardShader,
    BillboardSphereShader,
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

    colors = normalize_colors(colors)
    if colors.shape[0] == 1:
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
    indices = np.arange(pos.shape[0], dtype=np.uint32).reshape(-1, 3)

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


class BillboardActor(Mesh):
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

    def _wgpu_get_pick_info(self, pick_value):
        """
        Decode a billboard glyph index from GPU picking readback.

        Parameters
        ----------
        pick_value : int
            Packed 64-bit picking value written by the billboard shader.

        Returns
        -------
        dict
            The zero-based ``glyph_index`` of the picked billboard.

        Notes
        -----
        The low 20 bits identify the object; the next 26 identify the glyph.
        Shifting right by 20 removes the object ID. ``(1 << 26) - 1`` has
        exactly 26 low bits set, so the mask keeps only the glyph ID.
        The renderer resolves the object separately. Mesh face indices and
        barycentric coordinates do not apply to billboard picking.

        The shaders carry glyph IDs between stages as two 13-bit float
        components, preserving IDs that a single float32 cannot represent.
        All vertices of a glyph carry the same pair. The fragment shader
        rounds and reconstructs the integer before packing the pick value,
        so this decoder receives the integer field, not the float pair.
        """
        return {"glyph_index": (int(pick_value) >> 20) & ((1 << 26) - 1)}

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


Billboard = BillboardActor


def _max_ellipsoids_per_chunk(*, max_buffer_size, max_storage_buffer_binding_size):
    """
    Return the capacity of the largest per-glyph buffer and pick index.

    Parameters
    ----------
    max_buffer_size : int
        Maximum size of an individual GPU buffer, in bytes.
    max_storage_buffer_binding_size : int
        Maximum size of a storage buffer binding, in bytes.

    Returns
    -------
    int
        Maximum glyph count allowed by both buffer limits and 26-bit picking.

    Raises
    ------
    ValueError
        If the buffer limits cannot hold one ellipsoid.
    """
    capacity = min(
        max_buffer_size // 48,
        max_storage_buffer_binding_size // 48,
        1 << 26,
    )
    if capacity < 1:
        raise ValueError("Device buffer limits cannot hold one ellipsoid.")
    return capacity


class _BillboardEllipsoid(Mesh):
    """
    Render ellipsoids from one compact parameter record per glyph.

    Parameters
    ----------
    centers : array-like, shape (N, 3) or (3,)
        Model-space centers of the ellipsoids.
    orientation_matrices : array-like, optional
        Orthonormal matrices with axes as columns, shaped (3, 3), (1, 3, 3),
        or (N, 3, 3). A single matrix is broadcast to all glyphs.
    lengths : array-like, optional
        Nonnegative semi-axis lengths, shaped (3,), (1, 3), or (N, 3).
        A single triple is broadcast; zero-axis glyphs are not rendered.
    colors : array-like or str, optional
        One RGB/RGBA color or one per glyph. Hex strings are also accepted.
    opacity : float, optional
        Material opacity multiplier applied to each color's alpha.
    enable_picking : bool, optional
        Whether retained surface fragments write glyph picking information.
    """

    def __init__(
        self,
        centers,
        *,
        orientation_matrices=None,
        lengths=(4, 2, 2),
        colors=(1, 0, 0),
        opacity=None,
        enable_picking=True,
    ):
        super().__init__()
        self.local.state_basis = "matrix"
        centers = np.asarray(centers)
        if centers.shape == (3,):
            centers = centers.reshape(1, 3)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("Centers must be (N, 3) array")
        count = len(centers)

        orientations = np.asarray(
            np.eye(3) if orientation_matrices is None else orientation_matrices
        )
        if orientations.shape == (3, 3):
            orientations = orientations[np.newaxis]
        if (
            orientations.ndim != 3
            or orientations.shape[1:] != (3, 3)
            or orientations.shape[0] not in (1, count)
        ):
            raise ValueError("Axes must be (3, 3), (1, 3, 3), or (N, 3, 3) array")
        orientations = np.broadcast_to(orientations, (count, 3, 3))

        lengths = np.asarray(lengths)
        if lengths.shape == (3,):
            lengths = lengths[np.newaxis]
        if (
            lengths.ndim != 2
            or lengths.shape[1] != 3
            or lengths.shape[0] not in (1, count)
        ):
            raise ValueError("Lengths must be (3,), (1, 3), or (N, 3) array")
        lengths = np.broadcast_to(lengths, (count, 3))
        if not np.isfinite(lengths).all() or np.any(lengths < 0):
            raise ValueError("Lengths must be finite and nonnegative.")
        active = np.all(lengths > 0, axis=1)
        if not np.isfinite(centers).all() or np.any(
            np.abs(centers) > np.finfo(np.float32).max
        ):
            raise ValueError("Centers must be finite and fit in float32 buffers.")
        if np.any(active & ~np.isfinite(orientations).all(axis=(1, 2))):
            raise ValueError("Active orientation matrices must be finite.")
        for column in range(3):
            column_scale = np.max(np.abs(orientations[:, :, column]), axis=1).astype(
                np.float64, copy=False
            )
            with np.errstate(over="ignore", invalid="ignore"):
                overflow = column_scale * lengths[:, column] > np.finfo(np.float32).max
            if np.any(active & overflow):
                raise ValueError("Ellipsoid parameters must fit in float32 buffers.")

        if (
            isinstance(colors, np.ndarray)
            and colors.dtype == np.float32
            and colors.ndim in (1, 2)
            and colors.shape[-1] in (3, 4)
            and not colors.max(initial=0) > 1
        ):
            colors = colors.reshape(-1, colors.shape[-1])
        else:
            colors = normalize_colors(colors)
        if (
            colors.ndim != 2
            or colors.shape[1] not in (3, 4)
            or colors.shape[0] not in (1, count)
        ):
            raise ValueError("Colors must contain one color or one per ellipsoid.")
        colors = np.broadcast_to(colors, (count, colors.shape[1]))
        if np.any(active & ~np.isfinite(colors).all(axis=1)):
            raise ValueError("Active colors must be finite.")
        opacity = validate_opacity(opacity)

        limits = gfx_wgpu.get_shared().device.limits
        capacity = _max_ellipsoids_per_chunk(
            max_buffer_size=limits["max-buffer-size"],
            max_storage_buffer_binding_size=limits["max-storage-buffer-binding-size"],
        )
        if count > capacity:
            raise ValueError("Ellipsoid count exceeds the device buffer limit.")

        # Keep one backing record for empty actors; no instances use it.
        backing_count = max(count, 1)
        positions = (
            np.ascontiguousarray(centers, dtype=np.float32)
            if count
            else np.zeros((1, 3), dtype=np.float32)
        )
        rgba = np.zeros((backing_count, 4), dtype=np.float32)
        axes = np.zeros((backing_count, 3, 4), dtype=np.float32)
        np.copyto(
            rgba[:count, : colors.shape[1]],
            colors,
            where=active[:, np.newaxis],
        )
        if colors.shape[1] == 3:
            rgba[:count, 3] = 1
        np.multiply(
            orientations.swapaxes(1, 2),
            lengths[:, :, np.newaxis],
            out=axes[:count, :, :3],
            where=active[:, np.newaxis, np.newaxis],
            dtype=np.result_type(orientations.dtype, lengths.dtype, np.float32),
        )

        geometry = buffer_to_geometry(
            positions=positions,
            colors=rgba,
            ellipsoid_axes=axes.reshape(-1, 4),
        )
        material = BillboardEllipsoidMaterial(
            color_mode="vertex",
            opacity=opacity,
            pick_write=enable_picking,
            side="both",
        )
        self.geometry = geometry
        self.material = material
        self.glyph_count = count

    def _wgpu_get_pick_info(self, pick_value):
        """
        Decode the integer-packed glyph field from GPU picking readback.

        Parameters
        ----------
        pick_value : int
            Packed 64-bit picking value written by the ellipsoid shader.

        Returns
        -------
        dict
            The zero-based ``glyph_index`` of the picked ellipsoid.

        Notes
        -----
        The shader transports the ID as two exact 13-bit float components
        between stages, then reconstructs the integer before packing it.
        The low 20 bits identify the object and the next 26 identify the glyph;
        this decoder receives that packed integer, not the float pair.
        """
        return {"glyph_index": (int(pick_value) >> 20) & ((1 << 26) - 1)}

    def get_bounding_box(self):
        """
        Return model-space bounds including each ellipsoid's semi-axes.

        Returns
        -------
        ndarray, shape (2, 3), or None
            Lower and upper bounding-box corners, or None for an empty actor.
        """
        if self.glyph_count == 0:
            return None
        centers = self.geometry.positions.data[: self.glyph_count]
        columns = self.geometry.ellipsoid_axes.data.reshape(-1, 3, 4)[:, :, :3]
        extents = np.einsum("ncr,ncr->nr", columns, columns, dtype=np.float64)
        np.sqrt(extents, out=extents)
        return np.stack(
            ((centers - extents).min(axis=0), (centers + extents).max(axis=0))
        )


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
    colors : str, tuple, list or ndarray, optional
        Per-billboard color. Accepts a hex string, RGB(A) in [0, 1], RGB(A)
        in [0, 255], or one such color per billboard.
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
    colors : str, tuple, list or ndarray, optional
        Color per sphere. Accepts a hex string, RGB(A) in [0, 1], RGB(A) in
        [0, 255], or one such color per sphere. Single colors are broadcast.
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


@register_wgpu_render_function(_BillboardEllipsoid, BillboardEllipsoidMaterial)
def register_billboard_ellipsoid_render_function(wobject):
    """
    Register the pipeline for compact ellipsoid impostors.

    Parameters
    ----------
    wobject : _BillboardEllipsoid
        Ellipsoid world object to bind to the shader pipeline.

    Returns
    -------
    tuple
        Tuple containing the configured BillboardEllipsoidShader instance.
    """
    return (BillboardEllipsoidShader(wobject),)
