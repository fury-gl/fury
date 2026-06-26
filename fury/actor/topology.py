"""Surface actors for FURY."""

import logging
import os

import numpy as np

from fury.actor import Group, create_mesh
from fury.colormap import normalize_colors
from fury.geometry import buffer_to_geometry
from fury.io import load_image_texture
from fury.material import _create_mesh_material, validate_opacity
from fury.utils import generate_planar_uvs, voxel_mesh_by_object


def surface(
    vertices,
    faces,
    *,
    material="phong",
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
        The material type for the surface mesh. Options are 'phong' and 'basic'. This
        option only works with colors is passed.
    colors : str, tuple, list or ndarray, optional
        A per-vertex ``(N, 3)``/``(N, 4)`` array, or a single color as a hex
        string, RGB(A) in [0, 1], or RGB(A) in [0, 255].
    texture : str, optional
        Path to the texture image file.
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
                material=material, mode="vertex", opacity=opacity
            )
        elif isinstance(colors, (tuple, list, np.ndarray)) and len(colors) == 3:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
                normals=normals.astype("float32") if normals is not None else None,
            )
            mat = _create_mesh_material(
                material=material, mode="auto", opacity=opacity, color=colors
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
            material=material, texture=tex, opacity=opacity, mode="auto"
        )
    else:
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"), indices=faces.astype("int32")
        )
        mat = _create_mesh_material(material=material, opacity=opacity)

    obj = create_mesh(geo, mat)
    return obj


def contour_from_volume(data, *, color=(1, 0, 0), opacity=0.5, material="phong"):
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
        The material type for the contour mesh. Options are 'phong' and 'basic'.

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
        )
        contours.add(surface_actor)

    return contours
