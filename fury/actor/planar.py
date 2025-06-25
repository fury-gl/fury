"""Planar actors for rendering 2D shapes in 3D space."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from fury.actor import actor_from_primitive
from fury.geometry import buffer_to_geometry, create_image, create_point, create_text
from fury.material import (
    _create_image_material,
    _create_points_material,
    _create_text_material,
)
import fury.primitive as fp


def square(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Create one or many squares with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Square positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the square.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the square in each dimension. If a single value is provided,
        the same size will be used for all squares.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the squares. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the squares should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated squares, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> square_actor = actor.square(centers=centers, colors=colors)
    >>> _ = scene.add(square_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_square()
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


def star(
    centers,
    *,
    dim=2,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Create one or many stars with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Star positions.
    dim : int, optional
        The dimensionality of the star (2D or 3D).
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the star.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the star in each dimension. If a single value is
        provided, the same size will be used for all stars.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the stars. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the stars should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated stars, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> star_actor = actor.star(centers=centers, colors=colors)
    >>> _ = scene.add(star_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_star(dim=dim)
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


def disk(
    centers,
    *,
    colors=(1.0, 1.0, 1.0),
    radii=0.5,
    sectors=36,
    scales=(1.0, 1.0, 1.0),
    directions=(0.0, 0.0, 0.0),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many disks with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Disk positions.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    radii : float or ndarray (N,) or tuple, optional
        The radius of the disks, single value applies to all disks,
        while an array specifies a radius for each disk individually.
    sectors : int, optional
        The number of divisions around the disk's circumference .
        Higher values produce smoother disk.
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the disks in each dimension. If a single value is provided,
        the same size will be used for all disks.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the disk.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the disk. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the disk should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated disks, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> disk_actor = actor.disk(centers=centers, colors=colors)
    >>> _ = scene.add(disk_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    vertices, faces = fp.prim_disk(radius=radii, sectors=sectors)
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


def triangle(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Create one or many triangles with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Triangle positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the triangle.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the triangle in each dimension. If a single value is provided,
        the same size will be used for all triangles.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the triangles. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the triangles should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated triangles, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> triangle_actor = actor.triangle(centers=centers, colors=colors)
    >>> _ = scene.add(triangle_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_triangle()
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


def point(
    centers,
    *,
    size=4.0,
    colors=(1.0, 0.0, 0.0),
    material="basic",
    map=None,
    aa=True,
    opacity=1.0,
    enable_picking=True,
):
    """Create one or many points with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        The positions of the points.
    size : float, optional
        The size (diameter) of the points in logical pixels.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    material : str, optional
        The material type for the points.
        Options are 'basic', 'gaussian'.
    map : TextureMap or Texture, optional
        The texture map specifying the color for each texture coordinate.
    aa : bool, optional
        Whether or not the points are anti-aliased in the shader.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    enable_picking : bool, optional
        Whether the points should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A point actor containing the generated points with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(1000, 3) * 10
    >>> colors = np.random.rand(1000, 3)
    >>> point_actor = actor.point(centers=centers, colors=colors)
    >>> _ = scene.add(point_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    geo = buffer_to_geometry(
        positions=centers.astype("float32"),
        colors=colors.astype("float32"),
    )

    mat = _create_points_material(
        size=size,
        material=material,
        map=map,
        aa=aa,
        opacity=opacity,
        enable_picking=enable_picking,
    )

    obj = create_point(geo, mat)
    return obj


def marker(
    centers,
    *,
    size=15,
    colors=(1.0, 0.0, 0.0),
    marker="circle",
    edge_color="black",
    edge_width=1.0,
    opacity=1.0,
    enable_picking=True,
):
    """Create one or many markers with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        The positions of the markers.
    size : float, optional
        The size (diameter) of the points in logical pixels.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    marker : str or MarkerShape, optional
        The shape of the marker.
        Options are "●": "circle", "+": "plus", "x": "cross", "♥": "heart",
        "✳": "asterix".
    edge_color : str or tuple or Color, optional
        The color of line marking the edge of the markers.
    edge_width : float, optional
        The width of the edge of the markers.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    enable_picking : bool, optional
        Whether the points should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A marker actor containing the generated markers with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(1000, 3) * 10
    >>> colors = np.random.rand(1000, 3)
    >>> marker_actor = actor.marker(centers=centers, colors=colors)
    >>> _ = scene.add(marker_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    geo = buffer_to_geometry(
        positions=centers.astype("float32"),
        colors=colors.astype("float32"),
    )

    mat = _create_points_material(
        material="marker",
        size=size,
        marker=marker,
        edge_color=edge_color,
        edge_width=edge_width,
        opacity=opacity,
        enable_picking=enable_picking,
    )

    obj = create_point(geo, mat)
    return obj


def text(
    text,
    *,
    colors=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    font_size=1.0,
    family="Arial",
    anchor="middle-center",
    max_width=0.0,
    line_height=1.2,
    text_align="start",
    outline_color=(0.0, 0.0, 0.0),
    outline_thickness=0.0,
    opacity=1.0,
):
    """Create text with different features.

    Parameters
    ----------
    text : str or list[str]
        The plain text to render.
        The text is split in one TextBlock per line,
        unless a list is given, in which case each (str) item becomes a TextBlock.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    position : tuple, optional
        The (x, y, z) coordinates to place the text in 3D space.
    font_size : float, optional
        The size of the font, in object coordinates or pixel screen coordinates.
    family : str, optional
        The name(s) of the font to prefer.
    anchor : str, optional
        The position of the origin of the text. Can be "top-left",
        "top-center", "top-right", "middle-left", "middle-center",
        "middle-right", "bottom-left", "bottom-center", "bottom-right".
    max_width : float, optional
        The maximum width of the text. Words are wrapped if necessary.
    line_height : float, optional
        A factor to scale the distance between lines. A value of 1 means the
        "native" font's line distance.
    text_align : str, optional
        The horizontal alignment of the text. Can be "start",
        "end", "left", "right", "center", "justify" or "justify_all".
        Text alignment is ignored for vertical text.
    outline_color : tuple, optional
        The color of the outline of the text.
    outline_thickness : float, optional
        A value indicating the relative width of the outline. Valid values are
        between 0.0 and 0.5.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    Actor
        A text actor containing the generated text with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> text_actor = actor.text(text='FURY')
    >>> _ = scene.add(text_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    mat = _create_text_material(
        color=colors,
        opacity=opacity,
        outline_color=outline_color,
        outline_thickness=outline_thickness,
    )

    obj = create_text(
        text=text,
        material=mat,
        font_size=font_size,
        family=family,
        anchor=anchor,
        max_width=max_width,
        line_height=line_height,
        text_align=text_align,
    )

    obj.local.position = position

    return obj


def image(
    image,
    *,
    position=(0.0, 0.0, 0.0),
    directions=(0.0, 0.0, 1.0),
    visible=True,
    clim=None,
    map=None,
    gamma=1.0,
    interpolation="nearest",
):
    """
    Visualize a 2D image from a NumPy array or image file.

    Parameters
    ----------
    image : str or ndarray
        The image input. Can be a file path (string) or a NumPy array.
    position : tuple, optional
        The position of the image in 3D space.
    directions : ndarray, shape (3,) or tuple (3,), optional
        The orientation vector of the image.
    visible : bool, optional
        Whether the image should be visible.
    clim : tuple, optional
        Contrast limits for image scaling.
    map : TextureMap or Texture, optional
        The texture map used to convert image values into color.
    gamma : float, optional
        Gamma correction to apply to the image.
        Must be greater than 0.
    interpolation : str, optional
        Interpolation method for rendering the image.
        Either 'nearest' or 'linear'.

    Returns
    -------
    ImageActor
        An image actor containing the rendered 2D image.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> image_data = np.random.rand(256, 256)
    >>> image_actor = actor.image(image=image_data)
    >>> _ = scene.add(image_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    mat = _create_image_material(
        clim=clim,
        map=map,
        gamma=gamma,
        interpolation=interpolation,
    )

    obj = create_image(
        image_input=image,
        material=mat,
        visible=visible,
    )

    if interpolation not in ["nearest", "linear"]:
        raise ValueError(
            f"Interpolation must be 'nearest' or 'linear', but got {interpolation}."
        )
    if position is None:
        position = (0.0, 0.0, 0.0)

    if isinstance(position, (list, tuple, np.ndarray)) and len(position) == 3:
        position = np.asarray(position, dtype=np.float32)

    else:
        raise ValueError(f"Position must have a length  of 3. Got {position}.")

    if isinstance(directions, (list, tuple, np.ndarray)) and len(directions) == 3:
        directions = np.asarray(directions, dtype=np.float32)
    else:
        raise ValueError(f"Directions must have a length of 3. Got {directions}.")

    obj.local.position = position

    default_normal = np.array([0, 0, 1])
    target_normal = np.asarray(directions)
    target_normal = target_normal / np.linalg.norm(target_normal)

    rotation_axis = np.cross(default_normal, target_normal)
    dot_product = np.dot(default_normal, target_normal)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rot = R.from_rotvec(rotation_angle * rotation_axis)
    else:
        rot = R.from_quat([0, 0, 0, 1])

    obj.local.rotation = rot.as_quat()
    return obj
