import numpy as np

from fury.geometry import buffer_to_geometry, create_mesh
from fury.material import _create_mesh_material
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
        The height of the cylinder. Default is 1.
    sectors: int, optional
        The number of divisions around the cylinder's circumference .
        Higher values produce smoother cylinders. Default is 36.
    radii : float or ndarray (N,) or tuple, optional
        The radius of the base of the cylinders, single value applies to all cylinders,
        while an array specifies a radius for each cylinder individually. Default:0.5.
    scales : int or ndarray (N, 3) or tuple (3,), optional
        Scaling factors for the cylinders in the (x, y, z) dimensions.
        Default is uniform scaling (1, 1, 1).
    directions : ndarray, shape (N, 3), optional
        The orientation vector of the cylinder.
    capped : bool, optional
        Whether to add caps (circular ends) to the cylinders. Default is True.
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
        the same size will be used for all tetrahedron.
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
        the same size will be used for all icosahedron.
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
        provided, the same size will be used for all rhombicuboctahedron.
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
        provided, the same size will be used for all triangular prism.
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
        provided, the same size will be used for all pentagonal prism.
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
        provided, the same size will be used for all octagonal prism.
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
        The total height of the arrow, including the shaft and tip. Default is 1.0.
    resolution : int, optional
        The number of divisions along the arrow's circular cross-sections.
        Higher values produce smoother arrows. Default is 10.
    tip_length : float, optional
        The length of the arrowhead tip relative to the total height. Default is 0.35.
    tip_radius : float, optional
        The radius of the arrowhead tip. Default is 0.1.
    shaft_radius : float, optional
        The radius of the arrow shaft. Default is 0.03.
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the arrow in each dimension. If a single value is
        provided, the same size will be used for all arrow.
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
    roundness=(1, 1),
    directions=(0, 0, 0),
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
        Default is (1,1).
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1].
    scales : int or ndarray (N,3) or tuple (3,), optional
        The size of the superquadric in each dimension. If a single value is
        provided, the same size will be used for all superquadric.
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
        The height of the cone. Default is 1.
    sectors: int, optional
        The number of divisions around the cones's circumference .
        Higher values produce smoother cones. Default is 10.
    radii : float or ndarray (N,) or tuple, optional
        The radius of the base of the cones, single value applies to all cones,
        while an array specifies a radius for each cone individually. Default:0.5.
    scales : int or ndarray (N, 3) or tuple (3,), optional
        Scaling factors for the cones in the (x, y, z) dimensions.
        Default is uniform scaling (1, 1, 1).
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
