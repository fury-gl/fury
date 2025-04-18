import numpy as np

from fury.geometry import (
    _generate_smooth_points,
    _tube_frame,
    buffer_to_geometry,
    create_line,
    create_mesh,
    create_point,
    create_text,
    line_buffer_separator,
)
from fury.material import (
    _create_line_material,
    _create_mesh_material,
    _create_points_material,
    _create_text_material,
)
import fury.primitive as fp


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
):
    """Build an actor from a primitive.

    Parameters
    ----------
    vertices : ndarray
        Vertices of the primitive.
    faces : ndarray
        Faces of the primitive.
    centers : ndarray, shape (N, 3)
        Box positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,)
        The size of the box in each dimension.  If a single value is provided,
        the same size will be used for all boxes.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the box.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the boxes. Options are 'phong' and 'basic'.
    smooth : bool, optional
        Whether to create a smooth sphere or a faceted sphere.
    enable_picking : bool, optional
        Whether the boxes should be pickable in a 3D scene.
    repeat_primitive : bool, optional
        Whether to repeat the primitive for each center. If False,
        only one instance of the primitive is created at the first center.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated boxes, with the specified
        material and properties

    """

    if repeat_primitive:
        res = fp.repeat_primitive(
            vertices,
            faces,
            directions=directions,
            centers=centers,
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

    geo = buffer_to_geometry(
        indices=big_faces.astype("int32"),
        positions=big_vertices.astype("float32"),
        texcoords=big_vertices.astype("float32"),
        colors=big_colors.astype("float32"),
    )

    mat = _create_mesh_material(
        material=material, enable_picking=enable_picking, flat_shading=not smooth
    )
    obj = create_mesh(geometry=geo, material=mat)
    obj.local.position = centers[0]
    obj.prim_count = prim_count
    return obj


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
    lines: list of ndarray of shape (P, 3) or ndarray of shape (N, P, 3)
        Lines points.
    colors: ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    opacity: float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    material: str, optional
        The material type for the lines. Options are 'basic', 'segment', 'arrow',
        'thin', and 'thin_segment'.
    enable_picking: bool, optional
        Whether the lines should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
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
    >>> scene.add(line_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()

    """

    lines_positions, lines_colors = line_buffer_separator(
        lines, color=colors, color_mode="auto"
    )

    geo = buffer_to_geometry(
        positions=lines_positions.astype("float32"),
        colors=lines_colors.astype("float32")
        if lines_colors is not None
        else np.empty_like(lines_positions),
    )

    mat = _create_line_material(
        material=material, enable_picking=enable_picking, mode="auto", opacity=opacity
    )

    obj = create_line(geometry=geo, material=mat)

    obj.local.position = lines_positions[0]

    obj.prim_count = len(lines)

    return obj


def sphere(
    centers,
    *,
    colors=(1, 0, 0),
    radii=1.0,
    phi=16,
    theta=16,
    opacity=None,
    material="phong",
    enable_picking=True,
    smooth=True,
):
    """
    Visualize one or many spheres with different colors and radii.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Spheres positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    radii : float or ndarray, shape (N,), optional
        Sphere radius. Can be a single value for all spheres or an array of
        radii for each sphere.
    phi : int, optional
        The number of segments in the longitude direction.
    theta : int, optional
        The number of segments in the latitude direction.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the spheres. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the spheres should be pickable in a 3D scene.
    smooth : bool, optional
        Whether to create a smooth sphere or a faceted sphere.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated spheres, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> radii = np.random.rand(5)
    >>> sphere_actor = actor.sphere(centers=centers, colors=colors, radii=radii)
    >>> scene.add(sphere_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    scales = radii
    directions = (1, 0, 0)

    vertices, faces = fp.prim_sphere(phi=phi, theta=theta)
    return actor_from_primitive(
        vertices,
        faces,
        centers=centers,
        colors=colors,
        scales=scales,
        directions=directions,
        opacity=opacity,
        material=material,
        smooth=smooth,
        enable_picking=enable_picking,
    )


def ellipsoid(
    centers,
    *,
    axes=None,
    lengths=(4, 3, 2),
    colors=(1, 0, 0),
    opacity=None,
    phi=16,
    theta=16,
    material="phong",
    enable_picking=True,
    smooth=True,
):
    """
    Create ellipsoid actor(s) with specified orientation and scaling.

    Parameters
    ----------
    centers : ndarray (N, 3)
        Centers of the ellipsoids
    axes : ndarray, shape (N, 3, 3) or (3, 3), optional
        Rotation matrices for ellipsoids. Each 3×3 matrix defines a local
        coordinate frame, where the columns represent the ellipsoid’s x, y, and z
        axes in world space. This determines the ellipsoid's orientation.
        Must be orthonormal and right-handed.
    lengths : ndarray (N, 3) or (3,) or tuple (3,), optional
        Scaling factors along each axis
    colors : array-like or tuple, optional
        RGB/RGBA colors for each ellipsoid
    opacity : float, optional
        Overall transparency (0-1)
    phi : int, optional
        Latitude divisions
    theta : int, optional
        Longitude divisions
    material : str, optional
        Surface material type
    enable_picking : bool, optional
        Allow mouse selection
    smooth : bool, optional
        Smooth shading

    Returns
    -------
    actor : fury.actor.Actor
        Ellipsoid actor with transformations applied

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.array([[0, 0, 0]])
    >>> lengths = np.array([[2, 1, 1]])
    >>> colors = np.array([[1, 0, 0]])
    >>> ellipsoid = actor.ellipsoid(centers=centers,
    ...                             lengths=lengths,
    ...                             colors=colors)
    >>> scene.add(ellipsoid)
    >>> window.show(scene)
    """

    centers = np.asarray(centers)

    if axes is None:
        axes = np.tile(np.eye(3), (centers.shape[0], 1, 1))

    axes = np.asarray(axes)
    lengths = np.asarray(lengths)

    if centers.ndim == 1:
        centers = centers.reshape(1, 3)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Centers must be (N, 3) array")
    if axes.ndim == 2:
        axes = np.tile(axes, (centers.shape[0], 1, 1))
    if axes.ndim != 3 or axes.shape[1:] != (3, 3):
        raise ValueError("Axes must be (N, 3, 3) array")
    if lengths.ndim == 1:
        lengths = lengths.reshape(1, 3)
    if lengths.ndim != 2 or lengths.shape[1] != 3:
        raise ValueError("Lengths must be (N, 3) array")
    if lengths.shape != centers.shape:
        raise ValueError("Lengths must match centers shape")

    base_verts, base_faces = fp.prim_sphere(phi=phi, theta=theta)

    base_verts = np.asarray(base_verts)
    base_faces = np.asarray(base_faces)

    if base_verts.ndim != 2 or base_verts.shape[1] != 3:
        raise ValueError(f"base_verts has unexpected shape {base_verts.shape}")

    if isinstance(colors, (list, tuple)):
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = np.tile(colors, (centers.shape[0], 1))

    all_vertices = []
    all_faces = []
    all_colors = []

    for i in range(len(centers)):
        center = centers[i]
        orientation = axes[i]
        scale = lengths[i]

        transform = orientation @ np.diag(scale)
        transformed = (transform @ base_verts.T).T + center

        all_vertices.append(transformed)
        all_faces.append(base_faces + i * len(base_verts))

        all_colors.append(np.tile(colors[i], (len(base_verts), 1)))

    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_colors = np.vstack(all_colors)

    return actor_from_primitive(
        centers=centers,
        vertices=all_vertices,
        faces=all_faces,
        colors=all_colors,
        opacity=opacity,
        material=material,
        smooth=smooth,
        enable_picking=enable_picking,
        repeat_primitive=False,
    )


def box(
    centers,
    *,
    directions=(1, 0, 0),
    colors=(1, 0, 0),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
    detailed=True,
):
    """Visualize one or many boxes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Box positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the box.
    colors : ndarray, shape (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the box in each dimension.  If a single value is provided,
        the same size will be used for all boxes.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the boxes. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the boxes should be pickable in a 3D scene.
    detailed : bool, optional
        Whether to create a detailed box with 24 vertices or a simple box with
        8 vertices.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated boxes, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> scales = np.random.rand(5, 3)
    >>> box_actor = actor.box(centers=centers, scales=scales)
    >>> scene.add(box_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_box(detailed=detailed)
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


def cylinder(
    centers,
    *,
    colors=(1, 1, 1),
    height=1,
    sectors=36,
    radii=0.5,
    scales=(1, 1, 1),
    directions=(0, 1, 0),
    capped=True,
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many cylinders with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        cylinder positions.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    height: float, optional
        The height of the cylinder.
    sectors: int, optional
        The number of divisions around the cylinder's circumference .
        Higher values produce smoother cylinders.
    radii : float or ndarray (N,) or tuple, optional
        The radius of the base of the cylinders, single value applies to all cylinders,
        while an array specifies a radius for each cylinder individually.
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the square in each dimension.  If a single value is provided,
        the same size will be used for all cylinders.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the cylinder.
    capped : bool, optional
        Whether to add caps (circular ends) to the cylinders.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the cylinders. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the cylinders should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated cylinders, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> cylinder_actor = actor.cylinder(centers=centers, colors=colors)
    >>> scene.add(cylinder_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    vertices, faces = fp.prim_cylinder(
        radius=radii, height=height, sectors=sectors, capped=capped
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
    """Visualize one or many squares with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        square positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the square.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the square in each dimension.  If a single value is provided,
        the same size will be used for all squares.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the squares. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the squares should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
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
    >>> scene.add(square_actor)
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


def frustum(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many frustums with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        frustum positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the frustum.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the frustum in each dimension.  If a single value is provided,
        the same size will be used for all frustums.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the frustums. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the frustums should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated frustums, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> frustum_actor = actor.frustum(centers=centers, colors=colors)
    >>> scene.add(frustum_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_frustum()
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


def tetrahedron(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many tetrahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        tetrahedron positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the tetrahedron.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the tetrahedron in each dimension.  If a single value is provided,
        the same size will be used for all tetrahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the tetrahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the tetrahedrons should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated tetrahedrons, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> tetrahedron_actor = actor.tetrahedron(centers=centers, colors=colors)
    >>> scene.add(tetrahedron_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_tetrahedron()
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


def icosahedron(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many icosahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        icosahedron positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the icosahedron.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the icosahedron in each dimension.  If a single value is provided,
        the same size will be used for all icosahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the icosahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the icosahedrons should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated icosahedrons, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> icosahedron_actor = actor.icosahedron(centers=centers, colors=colors)
    >>> scene.add(icosahedron_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_icosahedron()
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


def rhombicuboctahedron(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many rhombicuboctahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        rhombicuboctahedron positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the rhombicuboctahedron.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the rhombicuboctahedro in each dimension. If a single value is
        provided, the same size will be used for all rhombicuboctahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the rhombicuboctahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the rhombicuboctahedrons should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated rhombicuboctahedrons, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> rhombicuboctahedron_actor = actor.rhombicuboctahedron(
    centers=centers, colors=colors
    )
    >>> scene.add(rhombicuboctahedron_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_rhombicuboctahedron()
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


def triangularprism(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many triangular prism with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        triangular prism positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the triangular prism.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the triangular prism in each dimension. If a single value is
        provided, the same size will be used for all triangular prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the triangular prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the triangular prisms should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated triangular prisms, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> triangularprism_actor = actor.triangularprism(
    centers=centers, colors=colors)
    >>> scene.add(triangularprism_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_triangularprism()
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


def pentagonalprism(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many pentagonal prism with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        pentagonal prism positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the pentagonal prism.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the pentagonal prism in each dimension. If a single value is
        provided, the same size will be used for all pentagonal prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the pentagonal prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the pentagonal prisms should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated pentagonal prism, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> pentagonalprism_actor = actor.pentagonalprism(
    centers=centers, colors=colors)
    >>> scene.add(pentagonalprism_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_pentagonalprism()
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


def octagonalprism(
    centers,
    *,
    directions=(0, 0, 0),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many octagonal prism with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        octagonal prism positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the octagonal prism.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the octagonal prism in each dimension. If a single value is
        provided, the same size will be used for all octagonal prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the octagonal prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the octagonal prisms should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated octagonal prisms, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> octagonalprism_actor = actor.octagonalprism(
    centers=centers, colors=colors)
    >>> scene.add(octagonalprism_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_octagonalprism()
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
    """Visualize one or many arrows with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        arrow positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the arrow.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
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
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the arrow in each dimension. If a single value is
        provided, the same size will be used for all arrows.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the arrows. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the arrows should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
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
    >>> scene.add(arrow_actor)
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


def superquadric(
    centers,
    *,
    directions=(0, 0, 0),
    roundness=(1, 1),
    colors=(1, 1, 1),
    scales=(1, 1, 1),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many superquadric with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        superquadric positions.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the superquadric.
    roundness : tuple, optional
        parameters (Phi and Theta) that control the shape of the superquadric.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the superquadric in each dimension. If a single value is
        provided, the same size will be used for all superquadrics.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the superquadrics. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the superquadrics should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated superquadrics, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> superquadric_actor = actor.superquadric(centers=centers, colors=colors)
    >>> scene.add(superquadric_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """
    vertices, faces = fp.prim_superquadric(roundness=roundness)
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


def cone(
    centers,
    *,
    colors=(1, 1, 1),
    height=1,
    sectors=10,
    radii=0.5,
    scales=(1, 1, 1),
    directions=(0, 1, 0),
    opacity=None,
    material="phong",
    enable_picking=True,
):
    """Visualize one or many cones with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        cone positions.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    height: float, optional
        The height of the cone.
    sectors: int, optional
        The number of divisions around the cones's circumference .
        Higher values produce smoother cones.
    radii : float or ndarray (N,) or tuple, optional
        The radius of the base of the cones, single value applies to all cones,
        while an array specifies a radius for each cone individually.
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the square in each dimension.  If a single value is provided,
        the same size will be used for all cones.
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the cone.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the cones. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the cones should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated cones, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> cone_actor = actor.cone(centers=centers, colors=colors)
    >>> scene.add(cone_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()
    """

    vertices, faces = fp.prim_cone(radius=radii, height=height, sectors=sectors)
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
    """Visualize one or many star with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        star positions.
    dim : int, optional.
        The dimensionality of the star (2D or 3D).
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the star.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the star in each dimension. If a single value is
        provided, the same size will be used for all stars.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity
    material : str, optional
        The material type for the stars. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the stars should be pickable in a 3D scene.

    Returns
    -------
    mesh_actor : Actor
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
    >>> scene.add(star_actor)
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
    """Visualize one or many points with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3).
        The positions of the points.
    size : float
        The size (diameter) of the points in logical pixels.
    colors : ndarray (N,3) or (N,4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    material : str, optional
        The material type for the points.
        Options are 'basic', 'gaussian'.
    map : TextureMap | Texture
        The texture map specifying the color for each texture coordinate.
    aa : bool, optional
        Whether or not the points are anti-aliased in the shader.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    enable_picking : bool, optional
        Whether the points should be pickable in a 3D scene.

    Returns
    -------
    point_actor : Actor
        An point actor containing the generated points with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(1000, 3) * 10
    >>> colors = np.random.rand(1000, 3)
    >>> point_actor = actor.point(centers=centers, colors=colors)
    >>> scene.add(point_actor)
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
    """Visualize one or many marker with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3).
        The positions of the markers.
    size : float
        The size (diameter) of the points in logical pixels.
    colors : ndarray (N,3) or (N,4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    marker : str | MarkerShape
        The shape of the marker.
        Options are "●": "circle", "+": "plus", "x": "cross", "♥": "heart",
        "✳": "asterix".
    edge_color : str | tuple | Color
        The color of line marking the edge of the markers.
    edge_width : float, optional
        The width of the edge of the markers.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    enable_picking : bool, optional
        Whether the points should be pickable in a 3D scene.

    Returns
    -------
    marker_actor : Actor
        An marker actor containing the generated points with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(1000, 3) * 10
    >>> colors = np.random.rand(1000, 3)
    >>> marker_actor = actor.marker(centers=centers, colors=colors)
    >>> scene.add(marker_actor)
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
    """Visualize text with different features.

    Parameters
    ----------
    text : str | list[str]
        The plain text to render (optional).
        The text is split in one TextBlock per line,
        unless a list is given, in which case each (str) item become a TextBlock.
    colors : ndarray (N,3) or (N,4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
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
    text_actor : Actor
        An text actor containing the generated text with the specified material
        and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> text_actor = actor.text(text='FURY')
    >>> scene.add(text_actor)
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
    return obj


def axes(
    *,
    scale=(1.0, 1.0, 1.0),
    color_x=(1.0, 0.0, 0.0),
    color_y=(0.0, 1.0, 0.0),
    color_z=(0.0, 0.0, 1.0),
    opacity=1.0,
):
    """Visualize coordinate system axes using colored arrow.

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
    axes_actor: Actor
        An axes actor representing the coordinate axes with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> axes_actor = actor.axes()
    >>> scene.add(axes_actor)
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


def tube(
    points,
    *,
    opacity=1.0,
    colors=(1, 1, 1),
    radius=0.2,
    segments=8,
    end_caps=True,
    num_interpolation_points=10,
    material="phong",
    enable_picking=True,
):
    """
    Create a stream tube actor from a set of control points.

    Parameters
    ----------
    points : ndarray (N, 3)
        The control points defining the path of the tube.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    colors : ndarray (N,3) or (N,4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
        If a single color is provided, it will be applied to all points.
        If an array of colors is provided, each point will have its own color.
        (WIP)
    radius : float, optional
        The radius of the tube.
    segments : int, optional
        The number of segments around the circumference of the tube.
    end_caps : bool, optional
        Whether to add end caps to the tube.
    num_interpolation_points : int, optional
        The number of interpolation points to generate between each pair
        of original points to smooth the curve.
    material : str, optional
        The material type for the tube. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the tube should be pickable in a 3D scene.

    Returns
    -------
    tube_actor : Actor
        A mesh actor containing the generated tube with the specified

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> points = np.array([[0, 0, 0], [0, 1, 1], [2, 0, 2],
    ...                   [3, 1, 3], [4, 0, 4], [5, 1, 5]])
    >>> colors = (1, 0, 1)
    >>> tube_actor = actor.tube(points=points, colors=colors)
    >>> scene.add(tube_actor)
    >>> show_manager = window.ShowManager(scene=scene, size=(600, 600))
    >>> show_manager.start()

    """

    smooth_points = _generate_smooth_points(points, num_interpolation_points)

    vertices = []
    indices = []

    if len(smooth_points) < 2:
        raise ValueError("At least two points are required to create a tube.")

    tangents = []
    for i in range(len(smooth_points)):
        if i == 0:
            t = smooth_points[1] - smooth_points[0]
        elif i == len(smooth_points) - 1:
            t = smooth_points[-1] - smooth_points[-2]
        else:
            t = smooth_points[i + 1] - smooth_points[i - 1]
        norm = np.linalg.norm(t)
        tangents.append(
            t / norm
            if norm > 0
            else (tangents[-1] if len(tangents) > 0 else np.array([0.0, 0.0, 1.0]))
        )

    frames = []
    _tube_frame(frames, tangents)

    for i, point in enumerate(smooth_points):
        if i < len(frames):
            n, b = frames[i]
        else:
            n, b = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

        angle_step = 2 * np.pi / segments
        for seg in range(segments):
            angle = seg * angle_step
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            vertex = point + n * x + b * y
            vertices.append(vertex)

    for i in range(len(smooth_points) - 1):
        for j in range(segments):
            next_j = (j + 1) % segments
            idx0 = i * segments + j
            idx1 = i * segments + next_j
            idx2 = (i + 1) * segments + j
            idx3 = (i + 1) * segments + next_j
            indices.extend([idx0, idx2, idx1])
            indices.extend([idx1, idx2, idx3])

    if end_caps and len(smooth_points) > 1:
        start_center = smooth_points[0]
        start_frame = (
            frames[0] if len(frames) > 0 else (np.array([1, 0, 0]), np.array([0, 1, 0]))
        )

        start_cap_start_index = len(vertices)
        vertices.append(start_center)

        angle_step = 2 * np.pi / segments
        for seg in range(segments):
            angle = seg * angle_step
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            vertex = start_center + start_frame[0] * x + start_frame[1] * y
            vertices.append(vertex)

        for seg in range(segments):
            next_seg = (seg + 1) % segments
            indices.extend(
                [
                    start_cap_start_index,
                    start_cap_start_index + seg + 1,
                    start_cap_start_index + next_seg + 1,
                ]
            )

        end_center = smooth_points[-1]

        end_frame = (
            frames[-1]
            if len(frames) > 0
            else (np.array([1, 0, 0]), np.array([0, 1, 0]))
        )

        end_cap_start_index = len(vertices)
        vertices.append(end_center)

        angle_step = 2 * np.pi / segments
        for seg in range(segments):
            angle = seg * angle_step
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            vertex = end_center + end_frame[0] * x + end_frame[1] * y
            vertices.append(vertex)

        for seg in range(segments):
            next_seg = (seg + 1) % segments
            indices.extend(
                [
                    end_cap_start_index,
                    end_cap_start_index + seg + 1,
                    end_cap_start_index + next_seg + 1,
                ]
            )
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32).reshape(-1, 3)

    tube_actor = actor_from_primitive(
        centers=np.zeros((1, 3)),
        vertices=vertices,
        faces=indices,
        colors=colors,
        scales=(1, 1, 1),
        opacity=opacity,
        material=material,
        enable_picking=enable_picking,
        smooth=True,
    )

    return tube_actor
