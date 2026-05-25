"""Slicer actor for Fury."""

import numpy as np

from fury.actor import (
    Actor,
    Group,
    Mesh,
    Volume,
    set_group_opacity,
    set_group_visibility,
    show_slices,
)
from fury.geometry import buffer_to_geometry
from fury.lib import (
    Geometry,
    MeshPhongShader,
    Texture,
    VolumeSliceMaterial,
    WorldObject,
    gfx_wgpu,
    register_wgpu_render_function,
)
from fury.material import (
    SphGlyphMaterial,
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
    _create_vector_field_material,
    validate_opacity,
)
import fury.primitive as fp
from fury.shader import (
    SphGlyphComputeShader,
    VectorFieldArrowShader,
    VectorFieldComputeShader,
    VectorFieldShader,
    VectorFieldThinShader,
)
from fury.utils import (
    create_sh_basis_matrix,
    get_lmax,
    get_n_coeffs,
)


def data_slicer(
    data,
    *,
    value_range=None,
    opacity=1.0,
    interpolation="linear",
    visibility=(True, True, True),
    initial_slices=None,
    alpha_mode="auto",
    depth_write=False,
):
    """
    Visualize a 3D volume data as a slice.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z) or (X, Y, Z, 3)
        The 3D volume data to be sliced.
    value_range : tuple, optional
        The minimum and maximum values for the color mapping.
        If None, the range is determined from the data.
    opacity : float, optional
        The opacity of the slice. Takes values from 0 (fully transparent) to 1 (opaque).
    interpolation : str, optional
        The interpolation method for the slice. Options are 'linear' and 'nearest'.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.
    initial_slices : tuple, optional
        A tuple of three initial slice positions in the x, y, and z dimensions,
        respectively. If None, the slices are initialized to the middle of the volume.
    alpha_mode : str, optional
        The alpha mode for the material. Please see the below link for details:
        https://docs.pygfx.org/stable/_autosummary/materials/pygfx.materials.Material.html#pygfx.materials.Material.alpha_mode.
    depth_write : bool, optional
        Whether to write depth information for the material.

    Returns
    -------
    Group
        An actor containing the generated slice with the specified properties.
    """

    if value_range is None:
        value_range = (np.min(data), np.max(data))

    if visibility is None:
        visibility = (True, True, True)

    if data.ndim < 3 or data.ndim > 4:
        raise ValueError(
            "Input data must be 3-dimensional or "
            "4-dimensional with last dimension of size 3."
        )
    elif data.ndim == 4 and (data.shape[-1] != 3 and data.shape[-1] != 4):
        raise ValueError("Last dimension must be of size 3 or 4.")

    opacity = validate_opacity(opacity)
    data = data.astype(np.float32)

    data = np.swapaxes(data, 0, 2)

    data_shape = data.shape
    if initial_slices is None:
        initial_slices = (
            data_shape[2] // 2,
            data_shape[1] // 2,
            data_shape[0] // 2,
        )

    texture = Texture(data, dim=3)

    slices = []
    for dim in [0, 1, 2]:  # XYZ
        abcd = [0, 0, 0, 0]
        abcd[dim] = -1
        abcd[-1] = data_shape[2 - dim] // 2
        mat = VolumeSliceMaterial(
            abcd,
            clim=value_range,
            interpolation=interpolation,
            pick_write=True,
            alpha_mode=alpha_mode,
            depth_write=depth_write,
        )
        geo = Geometry(grid=texture)
        plane = Volume(geo, mat)
        slices.append(plane)

    obj = Group(name="Slicer")
    obj.add(*slices)
    set_group_visibility(obj, visibility)
    show_slices(obj, initial_slices)
    set_group_opacity(obj, opacity)

    return obj


class VectorField(WorldObject, Actor):
    """
    Class to visualize a vector field.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    cross_section : list or tuple, shape (3,), optional
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be ignored and complete field will be shown.
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.
    """

    def __init__(
        self,
        field,
        *,
        actor_type="thin_line",
        cross_section=None,
        colors=None,
        scales=1.0,
        opacity=1.0,
        thickness=1.0,
        visibility=None,
    ):
        """Initialize a vector field."""
        super().__init__()
        if not (field.ndim == 5 or field.ndim == 4):
            raise ValueError(
                "Field must be 5D or 4D, "
                f"but got {field.ndim}D with shape {field.shape}"
            )

        total_vectors = np.prod(field.shape[:-1])
        if field.shape[-1] != 3:
            raise ValueError(
                f"Field must have last dimension as 3, but got {field.shape[-1]}"
            )

        self.vectors = field.reshape(total_vectors, 3).astype(np.float32)
        self.field_shape = field.shape[:3]
        if field.ndim == 4:
            self.vectors_per_voxel = 1
        else:
            self.vectors_per_voxel = field.shape[3]

        if isinstance(scales, (int, float)):
            self.scales = np.full((total_vectors, 1), scales, dtype=np.float32)
        elif scales.shape != field.shape[:-1]:
            raise ValueError(
                "Scales must match the shape of the field (X, Y, Z, N) or (X, Y, Z),"
                f" but got {scales.shape}"
            )
        else:
            self.scales = scales.reshape(total_vectors, 1).astype(np.float32)

        pnts_per_vector = 2
        pts = np.zeros((total_vectors * pnts_per_vector, 3), dtype=np.float32)
        pts[0] = self.field_shape

        if colors is None:
            colors = np.asarray((0, 0, 0), dtype=np.float32)
        else:
            colors = np.asarray(colors, dtype=np.float32)

        colors = np.tile(colors, (total_vectors * pnts_per_vector, 1))
        self.geometry = buffer_to_geometry(positions=pts, colors=colors)
        self.material = _create_vector_field_material(
            (0, 0, 0),
            visibility=visibility,
            material=actor_type,
            thickness=thickness,
            opacity=opacity,
        )

        if cross_section is None:
            self.cross_section = np.asarray([-2, -2, -2], dtype=np.int32)
        else:
            self.cross_section = cross_section

    def get_bounding_box(self):
        """
        Get the bounding box of the vector field.

        Returns
        -------
        list
            A list containing two elements, each a list of three floats representing
            the minimum and maximum coordinates of the bounding box.
        """
        return [
            [0, 0, 0],
            [self.field_shape[0] - 1, self.field_shape[1] - 1, self.field_shape[2] - 1],
        ]

    @property
    def cross_section(self):
        """
        Get the cross section of the vector field.

        Returns
        -------
        ndarray
            The cross section of the vector field.
        """
        return self.material.cross_section

    @cross_section.setter
    def cross_section(self, value):
        """
        Set the cross section of the vector field.

        Parameters
        ----------
        value : {list, tuple, ndarray}
            The cross section in world-space coordinates.  When this actor is
            part of a chunked group its ``local.position`` offset is already
            baked in, so the caller always works in the same coordinate frame
            as the full field.
        """
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(
                "Cross section must be a list, tuple, or ndarray, "
                f"but got {type(value)}"
            )
        if len(value) != 3:
            raise ValueError(f"Cross section must have length 3, but got {len(value)}")
        value = np.asarray(value, dtype=np.float32)
        bounds = self.bounds if hasattr(self, "bounds") else self.get_bounding_box()
        world_min = np.asarray(bounds[0], dtype=np.float32)
        world_max = np.asarray(bounds[1], dtype=np.float32)
        value = np.maximum(world_min, value)
        value = np.minimum(world_max, value)
        self.material.cross_section = value.astype(np.int32)

    @property
    def visibility(self):
        """
        Get the visibility of the vector field.

        Returns
        -------
        tuple
            A tuple of three boolean values indicating the visibility of the slices
            in the x, y, and z dimensions, respectively.
        """
        return self.material.visibility

    @visibility.setter
    def visibility(self, value):
        """
        Set the visibility of the vector field.

        Parameters
        ----------
        value : tuple
            A tuple of three boolean values indicating the visibility of the slices
            in the x, y, and z dimensions, respectively.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("Visibility must be a tuple of three boolean values.")
        self.material.visibility = value


def _max_voxels_per_chunk(*, max_buffer_size, max_workgroups):
    """
    Compute the maximum number of voxels that fit in one VectorField chunk.

    Two independent device limits constrain how large a single VectorField
    actor may be:

    * **Storage-buffer binding size** – the largest individual buffer that the
      GPU accepts.  The bottleneck buffer holds two float32 (x, y, z) vertices
      per vector (start + end point), so it costs ``n_vectors × 2 × 3 × 4``
      bytes.
    * **Compute workgroup dispatch dimension** – the compute pass dispatches
      ``ceil(n_voxels / workgroup_size)`` workgroups along the X axis.  WebGPU
      limits this to ``max-compute-workgroups-per-dimension`` (typically
      65 535).

    Parameters
    ----------
    max_buffer_size : int
        Device limit ``max-storage-buffer-binding-size`` in bytes.
    max_workgroups : int
        Device limit ``max-compute-workgroups-per-dimension``.

    Returns
    -------
    int
        Maximum number of voxels per chunk that satisfies both limits.
    """
    _VECTOR_FIELD_WORKGROUP_SIZE = 64
    from_buffer = max_buffer_size // (2 * 3 * 4)
    from_dispatch = max_workgroups * _VECTOR_FIELD_WORKGROUP_SIZE
    return min(from_buffer, from_dispatch)


def _create_chunked_vector_field(
    field,
    *,
    group_name,
    scales,
    actor_params,
    chunk_actor_postprocess=None,
):
    """
    Create a VectorField group, chunking when required by device limits.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        Vector field data.
    group_name : str
        Name for the returned Group.
    scales : {float, ndarray}
        Scalar value or per-voxel scales array.
    actor_params : dict
        Keyword arguments passed to each VectorField actor.
    chunk_actor_postprocess : callable, optional
        Callback invoked as ``fn(actor, x_start)`` for each chunk actor after
        chunk positioning, used for chunk-specific updates.

    Returns
    -------
    Group
        A Group containing one or more VectorField actors.
    """
    wgpu_device = gfx_wgpu.get_shared().device
    max_buffer_size = wgpu_device.limits.get(
        "max-storage-buffer-binding-size", 256 * 1024 * 1024
    )
    max_workgroups = wgpu_device.limits.get(
        "max-compute-workgroups-per-dimension", 65535
    )

    total_voxels = int(np.prod(field.shape[:3]))
    max_voxels = _max_voxels_per_chunk(
        max_buffer_size=max_buffer_size,
        max_workgroups=max_workgroups,
    )
    group = Group(name=group_name)

    if total_voxels <= max_voxels:
        actor = VectorField(field, scales=scales, **actor_params)
        group.add(actor)
        return group

    voxels_per_x_slice = int(np.prod(field.shape[1:3]))
    chunk_x = max(1, max_voxels // voxels_per_x_slice)

    x_size = field.shape[0]
    n_chunks = int(np.ceil(x_size / chunk_x))

    for i in range(n_chunks):
        x_start = i * chunk_x
        x_end = min(x_start + chunk_x, x_size)
        chunk_field = field[x_start:x_end]

        if isinstance(scales, np.ndarray):
            chunk_scales = scales[x_start:x_end]
        else:
            chunk_scales = scales

        actor = VectorField(chunk_field, scales=chunk_scales, **actor_params)
        actor.local.position = [x_start, 0, 0]
        if chunk_actor_postprocess is not None:
            chunk_actor_postprocess(actor, x_start)
        group.add(actor)

    return group


def vector_field(
    field,
    *,
    actor_type="thin_line",
    colors=None,
    scales=1.0,
    opacity=1.0,
    thickness=1.0,
):
    """
    Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.

    Returns
    -------
    Group
        A Group of VectorField chunks.
    """
    actor_params = {
        "actor_type": actor_type,
        "colors": colors,
        "opacity": opacity,
        "thickness": thickness,
    }
    return _create_chunked_vector_field(
        field,
        group_name="VectorField",
        scales=scales,
        actor_params=actor_params,
    )


def vector_field_slicer(
    field,
    *,
    actor_type="thin_line",
    cross_section=None,
    colors=None,
    scales=1.0,
    opacity=1.0,
    thickness=1.0,
    visibility=(True, True, True),
):
    """
    Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    cross_section : list or tuple, shape (3,), optional
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be ignored and complete field will be shown.
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.

    Returns
    -------
    VectorField or Group
        A single VectorField when the data fits within the device limits,
        or a Group of VectorField chunks otherwise.
    """
    if cross_section is None:
        cross_section = np.asarray(field.shape[:3], dtype=np.int32) // 2

    cross_section = np.asarray(cross_section, dtype=np.int32)

    actor_params = {
        "actor_type": actor_type,
        "cross_section": cross_section,
        "colors": colors,
        "opacity": opacity,
        "thickness": thickness,
        "visibility": visibility,
    }
    return _create_chunked_vector_field(
        field,
        group_name="VectorFieldSlicer",
        scales=scales,
        actor_params=actor_params,
        chunk_actor_postprocess=lambda actor, _: setattr(
            actor, "cross_section", cross_section.copy()
        ),
    )


@register_wgpu_render_function(VectorField, VectorFieldThinLineMaterial)
def register_vector_field_thin_shaders(wobject):
    """
    Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldThinShader(wobject)
    return compute_shader, render_shader


@register_wgpu_render_function(VectorField, VectorFieldLineMaterial)
def register_vector_field_shaders(wobject):
    """
    Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldShader(wobject)
    return compute_shader, render_shader


@register_wgpu_render_function(VectorField, VectorFieldArrowMaterial)
def register_vector_field_arrow_shaders(wobject):
    """
    Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldArrowShader(wobject)
    return compute_shader, render_shader


class SphGlyph(Mesh):
    """
    Visualize a spherical harmonic glyph with different features.

    Parameters
    ----------
    coeffs : ndarray, shape (X, Y, Z, N)
        The spherical harmonics coefficients. X, Y, Z denotes the position and N
        represents the number of coefficients.
    sphere : tuple
        Vertices and faces of the sphere to use for the glyph.
    basis_type : str, optional
        The type of basis to use for the spherical harmonics.
        Options are 'standard', 'descoteaux07'.
    color_type : str, optional
        The type of color mapping to use for the spherical glyph.
        Options are 'sign' and 'orientation'.
    shininess : float, optional
        The shininess of the material for the spherical glyph.
    """

    def __init__(
        self,
        coeffs,
        sphere,
        *,
        basis_type="standard",
        color_type="sign",
        shininess=50,
    ):
        """Visualize a spherical harmonic glyph with different features."""

        super().__init__()

        if not isinstance(coeffs, np.ndarray):
            raise TypeError("The attribute 'coeffs' must be a numpy ndarray.")
        elif coeffs.ndim != 4:
            raise ValueError(
                (
                    "The attribute 'coeffs' must be a 4D numpy ndarray "
                    "with shape (X, Y, Z, N)."
                )
            )
        elif coeffs.shape[-1] < 1:
            raise ValueError(
                "The last dimension of 'coeffs' must be greater than 0, "
                f"but got {coeffs.shape[-1]}"
            )

        if not isinstance(sphere, tuple):
            raise TypeError(
                "The attribute 'sphere' must be a tuple containing vertices and faces."
            )
        elif (
            len(sphere) != 2
            or not isinstance(sphere[0], np.ndarray)
            or not isinstance(sphere[1], np.ndarray)
        ):
            raise TypeError(
                "The attribute 'sphere' must be a tuple containing two numpy ndarrays "
                "(vertices, faces)."
            )

        self.n_coeff = coeffs.shape[-1]
        self.data_shape = coeffs.shape[:3]
        self.basis_type = basis_type
        self._l_max = get_lmax(self.n_coeff, basis_type=basis_type)
        self.color_type = 0 if color_type == "sign" else 1

        vertices, faces = sphere[0], sphere[1]
        positions = np.tile(vertices, (np.prod(self.data_shape), 1)).astype(np.float32)
        positions[0] = np.asarray(self.data_shape)
        self.scaled_vertices = np.zeros_like(positions, dtype=np.float32)

        self.vertices_per_glyph = vertices.shape[0]
        self.faces_per_glyph = faces.shape[0]

        self.indices = faces.reshape(-1).astype(np.int32)
        indices = np.tile(faces, (np.prod(self.data_shape), 1)).astype(np.int32)

        self.radii = np.zeros((self.vertices_per_glyph,), dtype=np.float32)

        for i in range(0, indices.shape[0], faces.shape[0]):
            start = i
            end = start + faces.shape[0]
            indices[start:end] += (i // faces.shape[0]) * self.vertices_per_glyph

        self.geometry = buffer_to_geometry(
            positions=positions.astype("float32"),
            indices=indices.astype("int32"),
            colors=np.ones_like(positions, dtype="float32"),
            normals=np.zeros_like(positions).astype("float32"),
        )

        self.material = SphGlyphMaterial(
            n_coeffs=self.n_coeff,
            color_mode="vertex",
            flat_shading=False,
            shininess=shininess,
            specular="#494949",
            side="front",
        )

        B_mat = create_sh_basis_matrix(vertices, self._l_max)
        self.sh_coeff = coeffs.reshape(-1).astype("float32")
        self.sf_func = B_mat.reshape(-1).astype("float32")
        self.sphere = vertices.astype("float32")

    @property
    def l_max(self):
        """
        Get the maximum degree of the spherical harmonics.

        Returns
        -------
        int
            The maximum degree of the spherical harmonics used in the glyph.
        """
        return self._l_max

    @l_max.setter
    def l_max(self, value):
        """
        Set the maximum degree of the spherical harmonics.

        Parameters
        ----------
        value : int
            The maximum degree of the spherical harmonics to set.

        Raises
        ------
        ValueError
            If the provided value is not a positive integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("The attribute 'l_max' must be a positive integer.")
        self._l_max = value
        self.material.n_coeffs = get_n_coeffs(value, basis_type=self.basis_type)

    @property
    def scale(self):
        """
        Get the scale of the spherical glyph.

        Returns
        -------
        float
            The scale of the spherical glyph.
        """
        return self.material.scale

    @scale.setter
    def scale(self, value):
        """
        Set the scale of the spherical glyph.

        Parameters
        ----------
        value : float
            The scale of the spherical glyph to set.
        """
        self.material.scale = value


def sph_glyph(
    coeffs, *, sphere=None, basis_type="standard", color_type="sign", shininess=50
):
    """
    Visualize a spherical harmonic glyph with different features.

    Parameters
    ----------
    coeffs : ndarray, shape (X, Y, Z, N)
        The spherical harmonics coefficients. X, Y, Z denotes the position and N
        represents the number of coefficients.
    sphere : {str, tuple}, optional
        The name of the sphere to use or a tuple containing the phi and theta
        segments for a custom sphere.
        Available options for the named spheres:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'
    basis_type : str, optional
        The type of basis to use for the spherical harmonics.
        Options are 'standard', 'descoteaux07'.
    color_type : str, optional
        The type of color mapping to use for the spherical glyph.
        Options are 'sign' and 'orientation'.
    shininess : float, optional
        The shininess of the material for the spherical glyph.

    Returns
    -------
    SphGlyph
        A spherical glyph object.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("The attribute 'coeffs' must be a numpy ndarray.")
    elif coeffs.ndim != 4:
        raise ValueError(
            "The attribute 'coeffs' must be a 4D numpy ndarray with shape (X, Y, Z, N)."
        )

    if sphere is None:
        sphere = "symmetric362"

    if isinstance(sphere, str):
        sphere = fp.prim_sphere(name=sphere)
    elif (
        isinstance(sphere, tuple)
        and isinstance(sphere[0], int)
        and isinstance(sphere[1], int)
    ):
        sphere = fp.prim_sphere(gen_faces=True, phi=sphere[0], theta=sphere[1])
    else:
        raise TypeError(
            "The attribute 'sphere' must be a string or tuple containing two integers."
        )

    if not isinstance(basis_type, str):
        raise TypeError("The attribute 'basis_type' must be a string.")
    elif basis_type not in ["standard", "descoteaux07"]:
        raise ValueError(
            "The attribute 'basis_type' must be either 'standard' or 'descoteaux07'."
        )

    if not isinstance(color_type, str):
        raise TypeError("The attribute 'color_type' must be a string.")
    if color_type not in ["sign", "orientation"]:
        raise ValueError(
            "The attribute 'color_type' must be either 'sign' or 'orientation'."
        )

    if not isinstance(shininess, (int, float)):
        raise TypeError("The attribute 'shininess' must be an integer or float.")

    obj = SphGlyph(
        coeffs=coeffs,
        sphere=sphere,
        basis_type=basis_type,
        color_type=color_type,
        shininess=shininess,
    )
    return obj


@register_wgpu_render_function(SphGlyph, SphGlyphMaterial)
def register_glyph_shaders(wobject):
    """
    Register Glyph shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = SphGlyphComputeShader(wobject)
    render_shader = MeshPhongShader(wobject)
    return compute_shader, render_shader
