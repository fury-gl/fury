"""Surface actors for FURY."""

import logging
import os

import numpy as np

from fury.actor import Group, create_mesh
from fury.colormap import normalize_colors
from fury.geometry import buffer_to_geometry
from fury.io import load_image, load_image_texture
from fury.lib import Texture, TextureMap
from fury.material import _create_mesh_material, validate_opacity
from fury.primitive import prim_sphere
from fury.utils import generate_planar_uvs, voxel_mesh_by_object


def surface(
    vertices,
    faces,
    *,
    material="phong",
    material_params=None,
    colors=None,
    texture=None,
    texture_axis="xy",
    texture_coords=None,
    normals=None,
    opacity=1.0,
):
    """
    Create a surface mesh actor from vertices and faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        The vertex positions of the surface mesh.
    faces : ndarray, shape (M, 3)
        The indices of the vertices that form each triangular face.
    material : str, optional
        The material type for the surface mesh. Options are 'phong', 'basic',
        'standard' and 'physical'. The last two are physically based (PBR)
        materials.
    material_params : dict, optional
        Extra properties forwarded to the material, e.g.
        ``{'metalness': 1.0, 'roughness': 0.2}`` for the PBR materials. See
        :func:`fury.material._create_mesh_material` for the supported keys.
    colors : str, tuple, list or ndarray, optional
        A per-vertex ``(N, 3)``/``(N, 4)`` array, or a single color as a hex
        string, RGB(A) in [0, 1], or RGB(A) in [0, 255].
    texture : str, optional
        Path to the texture image file. For spherical imagery, use
        :func:`texture_on_sphere`.
    texture_axis : str, optional
        The axis to generate UV coordinates for the texture. Options are 'xy', 'yz',
        and 'xz'. This option only works with texture is passed.
    texture_coords : ndarray, shape (N, 2), optional
        Predefined UV coordinates for the texture mapping. If not provided, they will
        be generated based on the `texture_axis`.
    normals : ndarray, shape (N, 3), optional
        The normal vectors for each vertex. If not provided, normals will be
        computed automatically.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    Mesh
        A mesh actor containing the generated surface with the specified properties.
    """
    geo = None
    mat = None

    material_params = material_params or {}
    opacity = validate_opacity(opacity)

    if colors is not None:
        if texture is not None:
            logging.warning("Texture will be ignored when colors are provided.")

        is_per_vertex = (
            isinstance(colors, np.ndarray) and colors.shape[0] == vertices.shape[0]
        )
        if not is_per_vertex:
            # A single color: normalize hex / [0, 255] / [0, 1] to RGB.
            colors = normalize_colors(colors)[0][:3]

        if isinstance(colors, np.ndarray) and colors.shape[0] == vertices.shape[0]:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
                colors=colors,
                normals=normals.astype("float32") if normals is not None else None,
            )
            mat = _create_mesh_material(
                material=material, mode="vertex", opacity=opacity, **material_params
            )
        elif isinstance(colors, (tuple, list, np.ndarray)) and len(colors) == 3:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
                normals=normals.astype("float32") if normals is not None else None,
            )
            mat = _create_mesh_material(
                material=material,
                mode="auto",
                opacity=opacity,
                color=colors,
                **material_params,
            )
        else:
            raise ValueError(
                "Colors must be either an ndarray with shape (N, 3) or (N, 4), "
                "or a tuple/list of length 3 for RGB colors."
            )
    elif texture is not None:
        if not os.path.exists(texture):
            raise FileNotFoundError(f"Texture file '{texture}' not found.")

        tex = load_image_texture(texture)
        if texture_coords is None:
            logging.warning(
                "texture option currently only supports planar projection,"
                " if the texture_coords are not provided, the plane can be provided"
                " by texture_axis parameter."
            )
            texture_coords = generate_planar_uvs(vertices, axis=texture_axis)
        elif (
            texture_coords.shape[0] != vertices.shape[0] or texture_coords.shape[1] != 2
        ):
            raise ValueError(
                "texture_coords must be an ndarray with shape (N, 2) "
                "where N is the number of vertices."
            )
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"),
            indices=faces.astype("int32"),
            texcoords=texture_coords.astype("float32"),
            normals=normals.astype("float32") if normals is not None else None,
        )
        mat = _create_mesh_material(
            material=material,
            texture=tex,
            opacity=opacity,
            mode="auto",
            **material_params,
        )
    else:
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"), indices=faces.astype("int32")
        )
        mat = _create_mesh_material(
            material=material, opacity=opacity, **material_params
        )

    obj = create_mesh(geo, mat)
    return obj


def texture_on_sphere(
    texture=None,
    *,
    theta=60,
    phi=60,
    interpolation="linear",
    material="phong",
    material_params=None,
    opacity=1.0,
):
    """
    Map an equirectangular image onto a smooth unit sphere.

    Parameters
    ----------
    texture : str, os.PathLike or ndarray, optional
        Local image filename or uint8 RGB/RGBA array of shape (H, W, 3) or
        (H, W, 4). None leaves the material color and environment in control.
    theta : int, optional
        Number of longitude segments, at least 3.
    phi : int, optional
        Number of latitude segments, at least 3.
    interpolation : str, optional
        Image sampling method, either 'linear' or 'nearest'.
    material : str, optional
        Mesh material: 'phong', 'basic', 'standard' or 'physical'.
    material_params : dict, optional
        Extra material properties, including PBR parameters. See
        :func:`fury.material._create_mesh_material`.
    opacity : float, optional
        Opacity from 0 (transparent) to 1 (opaque).

    Returns
    -------
    Mesh
        Origin-centered sphere of radius 1, with poles on the Y axis.

    Raises
    ------
    TypeError
        If the texture type or dtype is unsupported, or a resolution is not
        an integer.
    ValueError
        If an image is empty or not RGB/RGBA, a resolution is below 3, or
        interpolation, material properties or opacity are invalid.
    FileNotFoundError
        If the local image file does not exist.

    Notes
    -----
    U increases around Y from the seam at -Z through +Z at U=0.5.
    V runs from 0 at +Y to 1 at -Y; image rows are not flipped.
    Separate UV endpoints, radial normals and pole-safe tangents avoid
    geometric shading seams. The image's left and right edges must still
    match for its content to appear continuous.

    Use the mesh's ``local`` transform to change its radius or placement.
    Smooth shading is used unless overridden in ``material_params``.

    Examples
    --------
    >>> import numpy as np
    >>> from fury import actor
    >>> image = np.full((32, 64, 3), 255, dtype=np.uint8)
    >>> globe = actor.texture_on_sphere(image)
    >>> globe.local.scale = 2.0
    """
    for name, resolution in (("theta", theta), ("phi", phi)):
        if isinstance(resolution, (bool, np.bool_)) or not isinstance(
            resolution, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
        if resolution < 3:
            raise ValueError(f"{name} must be at least 3.")
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest'.")

    texture_map = None
    if texture is not None:
        if isinstance(texture, (str, os.PathLike)):
            filename = os.fspath(texture)
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Texture file '{filename}' not found.")
            pixels = load_image(filename)
        elif isinstance(texture, np.ndarray):
            pixels = texture
        else:
            raise TypeError("texture must be a local filename or NumPy array.")
        if pixels.dtype != np.uint8:
            raise TypeError("texture must have dtype uint8.")
        if (
            pixels.ndim != 3
            or pixels.shape[2] not in (3, 4)
            or pixels.shape[0] == 0
            or pixels.shape[1] == 0
        ):
            raise ValueError("texture must have nonempty shape (H, W, 3) or (H, W, 4).")
        texture_map = TextureMap(
            Texture(pixels, dim=2, generate_mipmaps=True),
            filter=interpolation,
            wrap_s="repeat",
            wrap_t="clamp",
        )

    vertices, faces = prim_sphere(phi=phi, theta=theta)
    normals = (vertices / np.linalg.norm(vertices, axis=1, keepdims=True)).astype(
        np.float32
    )
    u = np.linspace(0, 1, theta + 1)
    texcoords = np.empty((phi + 1, theta + 1, 2), dtype=np.float32)
    texcoords[:, :, 0] = u
    texcoords[:, :, 1] = np.linspace(1, 0, phi + 1)[:, None]

    longitude = 2 * np.pi * u - np.pi
    tangents = np.empty((phi + 1, theta + 1, 4), dtype=np.float32)
    tangents[:, :, 0] = np.cos(longitude)
    tangents[:, :, 1] = 0
    tangents[:, :, 2] = -np.sin(longitude)
    # cross(normal, tangent) points north; image V increases southward.
    tangents[:, :, 3] = -1
    geometry = buffer_to_geometry(
        positions=vertices.astype(np.float32),
        indices=faces,
        normals=normals,
        texcoords=texcoords.reshape(-1, 2),
        tangents=tangents.reshape(-1, 4),
    )
    params = {"flat_shading": False, **(material_params or {})}
    mesh_material = _create_mesh_material(
        material=material,
        texture=texture_map,
        mode="auto",
        opacity=opacity,
        **params,
    )
    return create_mesh(geometry, mesh_material)


def contour_from_volume(
    data, *, color=(1, 0, 0), opacity=0.5, material="phong", material_params=None
):
    """
    Generate surface actor from a binary ROI.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        An ROI file that will be binarized and displayed.
    color : str, tuple, list or ndarray, optional
        The output color of the contour. Accepts a hex string, RGB(A) in
        [0, 1], or RGB(A) in [0, 255].
    opacity : float, optional
        The opacity of the contour.
        Takes values from 0 (fully transparent) to 1 (opaque).
    material : str, optional
        The material type for the contour mesh. Options are 'phong', 'basic',
        'standard' and 'physical'. The last two are physically based (PBR)
        materials.
    material_params : dict, optional
        Extra properties forwarded to the material, e.g.
        ``{'metalness': 1.0, 'roughness': 0.2}`` for the PBR materials. See
        :func:`fury.material._create_mesh_material` for the supported keys.

    Returns
    -------
    Group
        A group of actors containing the generated contours from the volume data.
    """
    if color is None:
        raise ValueError("Color must be a tuple of three values (R, G, B).")
    color = normalize_colors(color)[0][:3]

    surface_data = voxel_mesh_by_object(data, connectivity=1)

    contours = Group()

    for surf in surface_data.values():
        surface_actor = surface(
            surf["verts"],
            surf["faces"],
            colors=color,
            opacity=opacity,
            material=material,
            material_params=material_params,
        )
        contours.add(surface_actor)

    return contours
