import numpy as np

from fury.geometry import (
    buffer_to_geometry,
    create_line,
    create_mesh,
    create_point,
    create_text,
    line_buffer_separator,
)
from fury.lib import WorldObject, register_wgpu_render_function
from fury.material import (
    VectorFieldMaterial,
    _create_line_material,
    _create_mesh_material,
    _create_points_material,
    _create_text_material,
)
import fury.primitive as fp
from fury.shader import VectorFieldComputeShader, VectorFieldShader


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

    Returns
    -------
    mesh_actor : Actor
        A mesh actor containing the generated boxes, with the specified
        material and properties

    """
    res = fp.repeat_primitive(
        vertices,
        faces,
        directions=directions,
        centers=centers,
        colors=colors,
        scales=scales,
    )
    big_vertices, big_faces, big_colors, _ = res

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

    if lines_colors is None:
        material_mode = "auto"
        material_colors = None
    else:
        material_mode = "vertex"
        material_colors = lines_colors

    mat = _create_line_material(
        material=material,
        enable_picking=enable_picking,
        mode=material_mode,
        opacity=opacity,
        color=material_colors,
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


class VectorField(WorldObject):
    def __init__(
        self, field, *, cross_section=None, colors=None, scales=1.0, opacity=1.0
    ):
        """Initialize a vector field.

        Parameters
        ----------
        field : ndarray, shape (X, Y, Z, N, 3) or (X, Y, Z, 3)
            The vector field data, where X, Y, Z represent the position in 3D,
            N is the number of vectors per voxel, and 3 represents the vector
        colors : tuple, optional
            Color for the vectors. If None, the color will used from the orientation.
        scales: float or ndarray, shape(X, Y, Z, N) or (X, Y, Z) optional
            Scale factor for the vectors. If ndarray, it should match the shape of the
            field.
        opacity : float, optional
            Takes values from 0 (fully transparent) to 1 (opaque).
        """

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

        if cross_section is None:
            cross_section = np.asarray([-1, -1, -1], dtype=np.int32)

        pnts_per_vector = 2
        pts = np.zeros((total_vectors * pnts_per_vector, 3), dtype=np.float32)
        pts[0] = self.field_shape

        if colors is None:
            colors = np.asarray((0, 0, 0), dtype=np.float32)
        else:
            colors = np.asarray(colors, dtype=np.float32)

        colors = np.tile(colors, (total_vectors * pnts_per_vector, 1))
        geometry = buffer_to_geometry(positions=pts, colors=colors)
        material = VectorFieldMaterial(cross_section, opacity=opacity)

        super().__init__(geometry=geometry, material=material)


def vector_field(field, *, colors=None, scales=1.0, opacity=1.0):
    """Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape (X, Y, Z, N, 3) or (X, Y, Z, 3)
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales: float or ndarray, shape(X, Y, Z, N) or (X, Y, Z) optional
            Scale factor for the vectors. If ndarray, it should match the shape of the
            field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    VectorField
        A vector field object.
    """

    obj = VectorField(field, colors=colors, scales=scales, opacity=opacity)
    return obj


def vector_field_slicer(
    field, *, cross_section=None, colors=None, scales=1.0, opacity=1.0
):
    """Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape (X, Y, Z, N, 3) or (X, Y, Z, 3)
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    cross_section : list or tuple, shape (3,)
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be set to the center of the field.
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales: float or ndarray, shape(X, Y, Z, N) or (X, Y, Z) optional
            Scale factor for the vectors. If ndarray, it should match the shape of the
            field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    VectorField
        A vector field object.
    """

    if cross_section is None:
        cross_section = np.asarray(field.shape[:3], dtype=np.int32) // 2

    obj = VectorField(
        field,
        cross_section=cross_section,
        colors=colors,
        scales=scales,
        opacity=opacity,
    )
    return obj


@register_wgpu_render_function(VectorField, VectorFieldMaterial)
def register_peaks_shaders(wobject):
    """Register PeaksActor shaders."""
    return VectorFieldComputeShader(wobject), VectorFieldShader(wobject)
