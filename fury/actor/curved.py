"""Curved primitives actors."""

import numpy as np

from fury.actor import actor_from_primitive
import fury.primitive as fp


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
    """Create one or many spheres with different colors and radii.

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
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the spheres. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the spheres should be pickable in a 3D scene.
    smooth : bool, optional
        Whether to create a smooth sphere or a faceted sphere.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(sphere_actor)
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
    orientation_matrices=None,
    lengths=(4, 2, 2),
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
        Centers of the ellipsoids.
    orientation_matrices : ndarray, shape (N, 3, 3) or (3, 3), optional
        Orthonormal rotation matrices defining the orientation of each ellipsoid.
        Each 3×3 matrix represents a local coordinate frame, with columns
        corresponding to the ellipsoid’s x-, y-, and z-axes in world coordinates.
        Must be right-handed and orthonormal. If a single (3, 3) matrix is
        provided, it is broadcast to all ellipsoids.
    lengths : ndarray (N, 3) or (3,) or tuple (3,), optional
        Scaling factors along each axis.
    colors : array-like or tuple, optional
        RGB/RGBA colors for each ellipsoid.
    opacity : float, optional
        Opacity of the ellipsoids. Takes values from 0 (fully transparent) to
        1 (opaque). If both `opacity` and RGBA are provided, the final alpha
        will be: final_alpha = alpha_in_RGBA * opacity.
    phi : int, optional
        The number of segments in the longitude direction.
    theta : int, optional
        The number of segments in the latitude direction.
    material : str, optional
        The material type for the ellipsoids. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Allow picking of the ellipsoids in a 3D scene.
    smooth : bool, optional
        Whether to create a smooth ellipsoid or a faceted ellipsoid.

    Returns
    -------
    Actor
        A mesh actor containing the generated ellipsoids.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> from fury import actor, window
    >>> centers = np.array([[0, 0, 0]])
    >>> lengths = np.array([[2, 1, 1]])
    >>> colors = np.array([[1, 0, 0]])
    >>> ellipsoid = actor.ellipsoid(centers=centers, lengths=lengths, colors=colors)
    >>> window.show([ellipsoid])
    """

    centers = np.asarray(centers)

    if orientation_matrices is None:
        orientation_matrices = np.tile(np.eye(3), (centers.shape[0], 1, 1))

    orientation_matrices = np.asarray(orientation_matrices)
    lengths = np.asarray(lengths)
    colors = np.asarray(colors)

    if centers.ndim == 1:
        centers = centers.reshape(1, 3)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Centers must be (N, 3) array")
    if orientation_matrices.ndim == 2:
        orientation_matrices = np.tile(orientation_matrices, (centers.shape[0], 1, 1))
    if orientation_matrices.ndim != 3 or orientation_matrices.shape[1:] != (3, 3):
        raise ValueError("Axes must be (N, 3, 3) array")
    if lengths.ndim == 1:
        lengths = lengths.reshape(1, 3)
    if lengths.ndim != 2 or lengths.shape[1] != 3:
        raise ValueError("Lengths must be (N, 3) array")
    if lengths.size == 3:
        lengths = np.tile(lengths.reshape(1, -1), (centers.shape[0], 1))
    if lengths.shape != centers.shape:
        raise ValueError("Lengths must match centers shape")
    if colors.size == 3 or colors.size == 4:
        colors = np.tile(colors.reshape(1, -1), (centers.shape[0], 1))

    base_verts, base_faces = fp.prim_sphere(phi=phi, theta=theta)

    base_verts = np.asarray(base_verts)
    base_faces = np.asarray(base_faces)

    if base_verts.ndim != 2 or base_verts.shape[1] != 3:
        raise ValueError(f"base_verts has unexpected shape {base_verts.shape}")

    if isinstance(colors, (list, tuple)):
        colors = np.asarray(colors)
        if colors.ndim == 1:
            colors = np.tile(colors, (centers.shape[0], 1))

    n_ellipsoids = centers.shape[0]
    n_verts = base_verts.shape[0]

    scaled_transforms = orientation_matrices * lengths[:, np.newaxis, :]

    transformed = (
        np.einsum("nij,mj->nmi", scaled_transforms, base_verts)
        + centers[:, np.newaxis, :]
    )
    all_vertices = transformed.reshape(-1, 3)
    all_faces = np.tile(base_faces, (n_ellipsoids, 1)) + (
        np.arange(n_ellipsoids)[:, None, None] * n_verts
    )
    all_faces = all_faces.reshape(-1, 3)
    all_colors = np.repeat(colors, n_verts, axis=0)

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
    """Create one or many cylinders with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cylinder positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    height : float, optional
        The height of the cylinder.
    sectors : int, optional
        The number of divisions around the cylinder's circumference.
        Higher values produce smoother cylinders.
    radii : float or ndarray, shape (N,) or tuple, optional
        The radius of the base of the cylinders. A single value applies to all
        cylinders,
        while an array specifies a radius for each cylinder individually.
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the cylinder in each dimension. If a single value is provided,
        the same size will be used for all cylinders.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the cylinder.
    capped : bool, optional
        Whether to add caps (circular ends) to the cylinders.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the cylinders. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the cylinders should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(cylinder_actor)
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
    """Create one or many cones with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Cone positions.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    height : float, optional
        The height of the cone.
    sectors : int, optional
        The number of divisions around the cone's circumference.
        Higher values produce smoother cones.
    radii : float or ndarray, shape (N,) or tuple, optional
        The radius of the base of the cones. A single value applies to all cones,
        while an array specifies a radius for each cone individually.
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the cone in each dimension. If a single value is provided,
        the same size will be used for all cones.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the cone.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the cones. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the cones should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(cone_actor)
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
