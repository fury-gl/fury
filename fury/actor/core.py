# -*- coding: utf-8 -*-
"""Core actor functionality for FURY."""

from PIL import Image as PILImage
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from fury.actor import set_opacity
from fury.geometry import (
    buffer_to_geometry,
    line_buffer_separator,
)
from fury.lib import (
    Geometry,
    ImageBasicMaterial,
    MeshBasicMaterial,
    MeshPhongMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    TextMaterial,
    Texture,
    gfx,
)
from fury.material import _create_line_material, _create_mesh_material
import fury.primitive as fp
from fury.transform import rotate, scale, translate


class Actor:
    """Base Actor class for making APIs user-friendly."""

    def rotate(self, rotation):
        """Rotate the actor by the given rotation.

        Parameters
        ----------
        rotation : tuple
            Rotation angles (in degrees) around the x, y, and z axes.
        """
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (3,):
            raise ValueError("Rotation must contain three angles (degrees).")

        quaternion = self._euler_to_quaternion(np.radians(rotation))
        rotate(quaternion, actor=self)

    def translate(self, translation):
        """Translate the actor by the given translation vector.

        Parameters
        ----------
        translation : tuple
            Translation vector along the x, y, and z axes.
        """
        if not isinstance(translation, (list, tuple, np.ndarray)):
            raise ValueError("Translation must be a sequence of three values.")
        translation = np.asarray(translation, dtype=np.float32)
        if translation.shape != (3,):
            raise ValueError("Translation must contain three values.")
        translate(translation, actor=self)

    def scale(self, scales):
        """Scale the actor by the given scale factors.

        Parameters
        ----------
        scales : tuple or float
            Scale factors along the x, y, and z axes. If a single float
            is provided, uniform scaling is applied.
        """

        if isinstance(scales, (int, float)):
            scales = (scales, scales, scales)
        elif not isinstance(scales, (list, tuple, np.ndarray)):
            raise ValueError(
                "Scale must be a sequence of three values or a single float."
            )

        scales = np.asarray(scales, dtype=np.float32)
        if scales.shape != (3,):
            raise ValueError("Scale must contain three values.")
        scale(scales, actor=self)

    def transform(self, matrix):
        """Apply a transformation matrix to the actor.

        This transformation replaces any existing transformations.

        Parameters
        ----------
        matrix : ndarray, shape (4, 4)
            Transformation matrix to be applied to the actor.
        """

        if not isinstance(matrix, np.ndarray):
            raise ValueError("Transformation matrix must be a numpy array.")
        elif matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be of shape (4, 4).")
        self.local.matrix = matrix

    @property
    def opacity(self):
        """Get the opacity of the actor.

        Returns
        -------
        float
            Opacity value between 0 (fully transparent) and 1 (fully opaque).
        """
        if isinstance(self, Group):
            if len(self.children) == 0:
                return 1.0
            return self.children[0].material.opacity
        return self.material.opacity

    @opacity.setter
    def opacity(self, opacity):
        """Set the opacity of the actor.

        Parameters
        ----------
        opacity : float
            Opacity value between 0 (fully transparent) and 1 (fully opaque).
        """
        set_opacity(self, opacity)

    @staticmethod
    def _euler_to_quaternion(rotation):
        """Convert XYZ Euler angles (radians) to a quaternion.

        Parameters
        ----------
        rotation : tuple or ndarray
            Rotation angles (in radians) around the x, y, and z axes.

        Returns
        -------
        ndarray
            Quaternion representing the rotation.
        """

        return Rot.from_euler("xyz", rotation).as_quat()


class Mesh(gfx.Mesh, Actor):
    """Mesh actor class."""


class Points(gfx.Points, Actor):
    """Points actor class."""


class Line(gfx.Line, Actor):
    """Line actor class."""


class Text(gfx.Text, Actor):
    """Text actor class."""


class Image(gfx.Image, Actor):
    """Image actor class."""


class Volume(gfx.Volume, Actor):
    """Volume actor class."""


class Group(gfx.Group, Actor):
    """Group actor class."""


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
    wireframe=False,
    wireframe_thickness=1.0,
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
    wireframe : bool, optional
        Whether to render the mesh as a wireframe.
    wireframe_thickness : float, optional
        The thickness of the wireframe lines.

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

    is_transparent = False
    if isinstance(opacity, (int, float)) and opacity < 1.0:
        is_transparent = True
    elif big_colors.shape[1] == 4 and np.any(big_colors[:, 3] < 1.0):
        is_transparent = True

    geo = buffer_to_geometry(
        indices=big_faces.astype("int32"),
        positions=big_vertices.astype("float32"),
        texcoords=big_vertices.astype("float32"),
        colors=big_colors.astype("float32"),
    )

    mat = _create_mesh_material(
        material=material,
        enable_picking=enable_picking,
        flat_shading=not smooth,
        wireframe=wireframe,
        wireframe_thickness=wireframe_thickness,
        alpha_mode="weighted_blend" if is_transparent else "auto",
        depth_write=not is_transparent,
    )
    obj = create_mesh(geometry=geo, material=mat)
    if not repeat_primitive:
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


def create_axes_helper(
    *,
    labels=None,
    colors=None,
    thickness=2,
    center_disk_radius=0.11,
    endpoint_disk_radius=0.33,
    label_font_size=0.4,
):
    """Create actors composing a UI axes helper.

    This returns the helper group and related actor lists so callers can
    attach callbacks and place it in scene-specific coordinate systems.

    Parameters
    ----------
    labels : list of str, optional
        Labels for [-X, +X, -Y, +Y, -Z, +Z].
    colors : list of tuple, optional
        RGB colors for each axis endpoint.
    thickness : float, optional
        Thickness for endpoint lines.
    center_disk_radius : float, optional
        Radius of the center disk.
    endpoint_disk_radius : float, optional
        Radius of endpoint disks.
    label_font_size : float, optional
        Font size for endpoint labels.

    Returns
    -------
    dict
        A dictionary containing:
        - group
        - center_disk
        - disks
        - labels
        - lines
        - line_points
        - axis_vectors
    """
    from fury.actor.curved import streamlines
    from fury.actor.planar import disk, text

    if labels is None:
        labels = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    elif labels is not None and len(labels) != 6:
        raise ValueError(
            "labels must be a list of 6 strings for [-X, +X, -Y, +Y, -Z, +Z]."
        )

    if colors is None:
        colors = [
            (0.9, 0.3, 0.23),
            (0.9, 0.3, 0.23),
            (0.5, 0.7, 0),
            (0.5, 0.7, 0),
            (0, 0, 0.7),
            (0, 0, 0.7),
        ]
    elif colors is not None and len(colors) != 6:
        raise ValueError(
            "colors must be a list of 6 RGB tuples for [-X, +X, -Y, +Y, -Z, +Z]."
        )

    group = Group(name="Axes Helper")
    centers = [
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]

    center_disk = disk(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        radii=center_disk_radius,
        colors=(0.5, 0.5, 0.5),
        material="basic",
    )
    group.add(center_disk)

    disks = []
    labels_actors = []
    lines = []
    line_points = []

    for i, endpoint_center in enumerate(centers):
        disk_actor = disk(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            radii=endpoint_disk_radius,
            colors=colors[i],
            material="basic",
        )
        disk_actor.local.position = endpoint_center.tolist()

        label_actor = text(
            labels[i],
            position=endpoint_center.tolist(),
            font_size=label_font_size,
        )

        axis_dir = endpoint_center / np.linalg.norm(endpoint_center)
        line_start = (axis_dir * center_disk_radius).tolist()
        line_end = (endpoint_center - axis_dir * endpoint_disk_radius).tolist()
        line_actor = streamlines(
            [[line_start, line_end]],
            colors=[colors[i]],
            thickness=thickness,
            outline_thickness=0,
        )

        disks.append(disk_actor)
        labels_actors.append(label_actor)
        lines.append(line_actor)
        line_points.append((line_start, line_end))

        # Due to the convention of the z -ve is forward.
        axis_vectors = centers[:4] + centers[5:] + centers[4:5]

        group.add(disk_actor)
        group.add(label_actor)
        group.add(line_actor)

    return {
        "group": group,
        "center_disk": center_disk,
        "disks": disks,
        "labels": labels_actors,
        "lines": lines,
        "line_points": line_points,
        "axis_vectors": axis_vectors,
    }


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

    lines_positions, lines_colors = line_buffer_separator(lines, color=colors)

    geo = buffer_to_geometry(
        positions=lines_positions.astype("float32"),
        colors=lines_colors.astype("float32"),
    )

    mat = _create_line_material(
        material=material,
        enable_picking=enable_picking,
        mode="vertex",
        opacity=opacity,
    )

    obj = create_line(geometry=geo, material=mat)

    obj.local.position = lines_positions[0]

    obj.prim_count = len(lines)

    return obj
