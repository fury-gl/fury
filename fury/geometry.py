"""Geometry utilities for FURY."""

from PIL import Image as PILImage
import numpy as np

from fury.lib import (
    Geometry,
    Image,
    ImageBasicMaterial,
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
    Texture,
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
    if positions is None:
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


def create_image(image_input, material, **kwargs):
    """Create an image object.

    Parameters
    ----------
    image_input : str or np.ndarray, optional
        The image content.
    material : Material
        The material object.
    **kwargs : dict, optional
        Additional properties like position, visible, etc.

    Returns
    -------
    Image
        The image object.
    """
    if isinstance(image_input, str):
        image = np.flipud(np.array(PILImage.open(image_input)).astype(np.float32))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim not in (2, 3):
            raise ValueError("image_input must be a 2D or 3D NumPy array.")
        if image_input.ndim == 3 and image_input.shape[2] not in (1, 3, 4):
            raise ValueError("image_input must have 1, 3, or 4 channels.")
        image = image_input
    else:
        raise TypeError("image_input must be a file path (str) or a NumPy array.")

    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    if image.max() > 1.0 or image.min() < 0.0:
        if image.max() == image.min():
            raise ValueError("Cannot normalize an image with constant pixel values.")
        image = (image - image.min()) / (image.max() - image.min())

    if not isinstance(material, ImageBasicMaterial):
        raise TypeError("material must be an instance of ImageBasicMaterial.")

    image = Image(
        Geometry(grid=Texture(image.astype(np.float32), dim=2)), material=material
    )
    return image
