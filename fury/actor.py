import numpy as np

from fury.geometry import buffer_to_geometry, create_mesh, create_point
from fury.lib import Geometry, Group, Texture, Volume, VolumeSliceMaterial
from fury.material import (
    _create_mesh_material,
    _create_points_material,
    validate_opacity,
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


def slicer(
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
    slicer_actor : Group
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
            opacity=opacity,
            pick_write=True,
        )
        geo = Geometry(grid=texture)
        plane = Volume(geo, mat)
        slices.append(plane)

    class Slicer(Group):
        def __init__(
            self,
            *,
            visible=True,
        ):
            super().__init__(visible=visible, name="Slicer")

        def show_slices(self, position):
            """Show the slices at the specified position.

            Parameters
            ----------
            position : tuple
                A tuple containing the positions of the slices in the 3D space.
            """
            for i, child in enumerate(self.children):
                a, b, c, _ = child.material.plane
                child.material.plane = (a, b, c, position[i])

        def set_opacity(self, opacity):
            """Set the opacity of the slices.

            Parameters
            ----------
            opacity : float
                The opacity value to set for the slices,
                ranging from 0 (fully transparent) to 1 (opaque).
            """
            for child in self.children:
                child.material.opacity = opacity

        def set_visibility(self, visibility):
            """Set the visibility of the slices.

            Parameters
            ----------
            visibility : tuple
                A tuple of three boolean values indicating the visibility of the slices
                in the x, y, and z dimensions, respectively.
            """
            for i, child in enumerate(self.children):
                child.visible = visibility[i]

        def get_slices(self):
            """Get the current positions of the slices.

            Returns
            -------
            position : ndarray
                An array containing the current positions of the slices.
            """
            return np.asarray([child.material.plane[-1] for child in self.children])

    obj = Slicer()
    obj.add(*slices)
    obj.set_visibility(visibility)
    obj.show_slices(initial_slices)

    return obj
