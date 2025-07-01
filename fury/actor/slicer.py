"""Slicer actor for Fury."""

import numpy as np

from fury.geometry import buffer_to_geometry
from fury.lib import (
    Geometry,
    Group,
    Mesh,
    MeshPhongShader,
    Texture,
    Volume,
    VolumeSliceMaterial,
    WorldObject,
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
    set_group_opacity,
    set_group_visibility,
    show_slices,
)


def data_slicer(
    data,
    *,
    value_range=None,
    opacity=1.0,
    interpolation="linear",
    visibility=(True, True, True),
    initial_slices=None,
):
    """Visualize a 3D volume data as a slice.

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
    elif data.ndim == 4 and data.shape[-1] != 3:
        raise ValueError("Last dimension must be of size 3.")

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


class VectorField(WorldObject):
    """Class to visualize a vector field.

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
        """Initialize a vector field.

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
        self.visibility = visibility
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
            material=actor_type,
            thickness=thickness,
            opacity=opacity,
        )

        if cross_section is None:
            self.cross_section = np.asarray([-2, -2, -2], dtype=np.int32)
        else:
            self.cross_section = cross_section

    @property
    def cross_section(self):
        """Get the cross section of the vector field.

        Returns
        -------
        ndarray
            The cross section of the vector field.
        """
        return self.material.cross_section

    @cross_section.setter
    def cross_section(self, value):
        """Set the cross section of the vector field.

        Parameters
        ----------
        value : {list, tuple, ndarray}
            The cross section dimensions to set.
        """
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(
                "Cross section must be a list, tuple, or ndarray, "
                f"but got {type(value)}"
            )
        if len(value) != 3:
            raise ValueError(f"Cross section must have length 3, but got {len(value)}")
        if self.visibility is None:
            self.material.cross_section = np.asarray([-2, -2, -2], dtype=np.int32)
            return
        value = np.asarray(value, dtype=np.int32)
        value = np.minimum(np.asarray(self.field_shape) - 1, value)
        value = np.maximum(value, np.zeros((3,), dtype=np.int32))
        value = np.where(self.visibility, value, -1)
        self.material.cross_section = value


def vector_field(
    field,
    *,
    actor_type="thin_line",
    colors=None,
    scales=1.0,
    opacity=1.0,
    thickness=1.0,
):
    """Visualize a vector field with different features.

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
    VectorField
        A vector field object.
    """

    obj = VectorField(
        field,
        actor_type=actor_type,
        colors=colors,
        scales=scales,
        opacity=opacity,
        thickness=thickness,
    )
    return obj


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
    """Visualize a vector field with different features.

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
    VectorField
        A vector field object.
    """

    if cross_section is None:
        cross_section = np.asarray(field.shape[:3], dtype=np.int32) // 2

    obj = VectorField(
        field,
        actor_type=actor_type,
        cross_section=cross_section,
        colors=colors,
        scales=scales,
        opacity=opacity,
        thickness=thickness,
        visibility=visibility,
    )
    return obj


@register_wgpu_render_function(VectorField, VectorFieldThinLineMaterial)
def register_vector_field_thin_shaders(wobject):
    """Register PeaksActor shaders.

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
    """Register PeaksActor shaders.

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
    """Register PeaksActor shaders.

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
    """Visualize a spherical harmonic glyph with different features.

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
        """Visualize a spherical harmonic glyph with different features.

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
        l_max = get_lmax(self.n_coeff, basis_type=basis_type)
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

        geo = buffer_to_geometry(
            positions=positions.astype("float32"),
            indices=indices.astype("int32"),
            colors=np.ones_like(positions, dtype="float32"),
            normals=np.zeros_like(positions).astype("float32"),
        )

        mat = SphGlyphMaterial(
            l_max=l_max,
            color_mode="vertex",
            flat_shading=False,
            shininess=shininess,
            specular="#494949",
            side="front",
        )

        B_mat = create_sh_basis_matrix(vertices, l_max)
        self.sh_coeff = coeffs.reshape(-1).astype("float32")
        self.sf_func = B_mat.reshape(-1).astype("float32")
        self.sphere = vertices.astype("float32")

        super().__init__(geometry=geo, material=mat)


def sph_glyph(
    coeffs, *, sphere=None, basis_type="standard", color_type="sign", shininess=50
):
    """Visualize a spherical harmonic glyph with different features.

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
    """Register Glyph shaders.

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
