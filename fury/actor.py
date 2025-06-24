"""Actor creation functions for various geometric primitives."""

import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from fury.geometry import (
    buffer_to_geometry,
    create_image,
    create_line,
    create_mesh,
    create_point,
    create_text,
    line_buffer_separator,
)
from fury.io import load_image_texture
from fury.lib import (
    Geometry,
    Group,
    Mesh,
    MeshBasicMaterial,
    MeshPhongShader,
    Texture,
    Volume,
    VolumeSliceMaterial,
    WorldObject,
    register_wgpu_render_function,
)
from fury.material import (
    SphGlyphMaterial,
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
    _create_image_material,
    _create_line_material,
    _create_mesh_material,
    _create_points_material,
    _create_text_material,
    _create_vector_field_material,
    validate_opacity,
)
import fury.primitive as fp
from fury.shader import (
    SphGlyphComputeShader,
    VectorFieldArrowShader,
    VectorFieldComputeShader,
    VectorFieldShader,
    VectorFieldThinShader,
)
from fury.utils import (
    create_sh_basis_matrix,
    generate_planar_uvs,
    get_lmax,
    set_group_opacity,
    set_group_visibility,
    show_slices,
)


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
    """Create one or many boxes with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Box positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the box.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the box in each dimension. If a single value is provided,
        the same size will be used for all boxes.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the boxes. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the boxes should be pickable in a 3D scene.
    detailed : bool, optional
        Whether to create a detailed box with 24 vertices or a simple box with
        8 vertices.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(box_actor)
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
    """Create one or many frustums with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Frustum positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the frustum.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the frustum in each dimension. If a single value is provided,
        the same size will be used for all frustums.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the frustums. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the frustums should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(frustum_actor)
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
    """Create one or many tetrahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Tetrahedron positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the tetrahedron.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the tetrahedron in each dimension. If a single value is provided,
        the same size will be used for all tetrahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the tetrahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the tetrahedrons should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(tetrahedron_actor)
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
    """Create one or many icosahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Icosahedron positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the icosahedron.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the icosahedron in each dimension. If a single value is provided,
        the same size will be used for all icosahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the icosahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the icosahedrons should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(icosahedron_actor)
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
    """Create one or many rhombicuboctahedrons with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Rhombicuboctahedron positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the rhombicuboctahedron.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the rhombicuboctahedron in each dimension. If a single value is
        provided, the same size will be used for all rhombicuboctahedrons.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the rhombicuboctahedrons. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the rhombicuboctahedrons should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    ...    centers=centers, colors=colors)
    >>> _ = scene.add(rhombicuboctahedron_actor)
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
    """Create one or many triangular prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Triangular prism positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the triangular prism.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the triangular prism in each dimension. If a single value is
        provided, the same size will be used for all triangular prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the triangular prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the triangular prisms should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated triangular prisms, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> triangularprism_actor = actor.triangularprism(centers=centers, colors=colors)
    >>> _ = scene.add(triangularprism_actor)
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
    """Create one or many pentagonal prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Pentagonal prism positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the pentagonal prism.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the pentagonal prism in each dimension. If a single value is
        provided, the same size will be used for all pentagonal prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the pentagonal prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the pentagonal prisms should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated pentagonal prisms, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> pentagonalprism_actor = actor.pentagonalprism(centers=centers, colors=colors)
    >>> _ = scene.add(pentagonalprism_actor)
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
    """Create one or many octagonal prisms with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Octagonal prism positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the octagonal prism.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the octagonal prism in each dimension. If a single value is
        provided, the same size will be used for all octagonal prisms.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the octagonal prisms. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the octagonal prisms should be pickable in a 3D scene.

    Returns
    -------
    Actor
        A mesh actor containing the generated octagonal prisms, with the specified
        material and properties.

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> scene = window.Scene()
    >>> centers = np.random.rand(5, 3) * 10
    >>> colors = np.random.rand(5, 3)
    >>> octagonalprism_actor = actor.octagonalprism(centers=centers, colors=colors)
    >>> _ = scene.add(octagonalprism_actor)
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
    """Create one or many superquadrics with different features.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
        Superquadric positions.
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the superquadric.
    roundness : tuple, optional
        Parameters (Phi and Theta) that control the shape of the superquadric.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA (for opacity) R, G, B, and A should be in the range [0, 1].
    scales : ndarray, shape (N, 3) or tuple (3,) or float, optional
        The size of the superquadric in each dimension. If a single value is
        provided, the same size will be used for all superquadrics.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
        If both `opacity` and RGBA are provided, the final alpha will be:
        final_alpha = alpha_in_RGBA * opacity.
    material : str, optional
        The material type for the superquadrics. Options are 'phong' and 'basic'.
    enable_picking : bool, optional
        Whether the superquadrics should be pickable in a 3D scene.

    Returns
    -------
    Actor
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
    >>> _ = scene.add(superquadric_actor)
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
    Group
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
            pick_write=True,
        )
        geo = Geometry(grid=texture)
        plane = Volume(geo, mat)
        slices.append(plane)

    obj = Group(name="Slicer")
    obj.add(*slices)
    set_group_visibility(obj, visibility)
    show_slices(obj, initial_slices)
    set_group_opacity(obj, opacity)

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


class VectorField(WorldObject):
    """Class to visualize a vector field.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    cross_section : list or tuple, shape (3,), optional
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be ignored and complete field will be shown.
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.
    """

    def __init__(
        self,
        field,
        *,
        actor_type="thin_line",
        cross_section=None,
        colors=None,
        scales=1.0,
        opacity=1.0,
        thickness=1.0,
        visibility=None,
    ):
        """Initialize a vector field.

        Parameters
        ----------
        field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
            The vector field data, where X, Y, Z represent the position in 3D,
            N is the number of vectors per voxel, and 3 represents the vector
        actor_type : str, optional
            The type of vector field visualization. Options are "thin_line",
            "line", and "arrow".
        cross_section : list or tuple, shape (3,), optional
            A list or tuple representing the cross section dimensions.
            If None, the cross section will be ignored and complete field will be shown.
        colors : tuple, optional
            Color for the vectors. If None, the color will used from the orientation.
        scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
            Scale factor for the vectors. If ndarray, it should match the shape of the
            field.
        opacity : float, optional
            Takes values from 0 (fully transparent) to 1 (opaque).
        thickness : float, optional
            The thickness of the lines in the vector field visualization.
            Only applicable for "line" and "arrow" types.
        visibility : tuple, optional
            A tuple of three boolean values indicating the visibility of the slices
            in the x, y, and z dimensions, respectively.
        """
        super().__init__()
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
        self.visibility = visibility
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

        pnts_per_vector = 2
        pts = np.zeros((total_vectors * pnts_per_vector, 3), dtype=np.float32)
        pts[0] = self.field_shape

        if colors is None:
            colors = np.asarray((0, 0, 0), dtype=np.float32)
        else:
            colors = np.asarray(colors, dtype=np.float32)

        colors = np.tile(colors, (total_vectors * pnts_per_vector, 1))
        self.geometry = buffer_to_geometry(positions=pts, colors=colors)
        self.material = _create_vector_field_material(
            (0, 0, 0),
            material=actor_type,
            thickness=thickness,
            opacity=opacity,
        )

        if cross_section is None:
            self.cross_section = np.asarray([-2, -2, -2], dtype=np.int32)
        else:
            self.cross_section = cross_section

    @property
    def cross_section(self):
        """Get the cross section of the vector field.

        Returns
        -------
        ndarray
            The cross section of the vector field.
        """
        return self.material.cross_section

    @cross_section.setter
    def cross_section(self, value):
        """Set the cross section of the vector field.

        Parameters
        ----------
        value : {list, tuple, ndarray}
            The cross section dimensions to set.
        """
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(
                "Cross section must be a list, tuple, or ndarray, "
                f"but got {type(value)}"
            )
        if len(value) != 3:
            raise ValueError(f"Cross section must have length 3, but got {len(value)}")
        if self.visibility is None:
            self.material.cross_section = np.asarray([-2, -2, -2], dtype=np.int32)
            return
        value = np.asarray(value, dtype=np.int32)
        value = np.minimum(np.asarray(self.field_shape) - 1, value)
        value = np.maximum(value, np.zeros((3,), dtype=np.int32))
        value = np.where(self.visibility, value, -1)
        self.material.cross_section = value


def vector_field(
    field,
    *,
    actor_type="thin_line",
    colors=None,
    scales=1.0,
    opacity=1.0,
    thickness=1.0,
):
    """Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.

    Returns
    -------
    VectorField
        A vector field object.
    """

    obj = VectorField(
        field,
        actor_type=actor_type,
        colors=colors,
        scales=scales,
        opacity=opacity,
        thickness=thickness,
    )
    return obj


def vector_field_slicer(
    field,
    *,
    actor_type="thin_line",
    cross_section=None,
    colors=None,
    scales=1.0,
    opacity=1.0,
    thickness=1.0,
    visibility=(True, True, True),
):
    """Visualize a vector field with different features.

    Parameters
    ----------
    field : ndarray, shape {(X, Y, Z, N, 3), (X, Y, Z, 3)}
        The vector field data, where X, Y, Z represent the position in 3D,
        N is the number of vectors per voxel, and 3 represents the vector
    actor_type : str, optional
        The type of vector field visualization. Options are "thin_line",
        "line", and "arrow".
    cross_section : list or tuple, shape (3,), optional
        A list or tuple representing the cross section dimensions.
        If None, the cross section will be ignored and complete field will be shown.
    colors : tuple, optional
        Color for the vectors. If None, the color will used from the orientation.
    scales : {float, ndarray}, shape {(X, Y, Z, N) or (X, Y, Z)}, optional
        Scale factor for the vectors. If ndarray, it should match the shape of the
        field.
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque).
    thickness : float, optional
        The thickness of the lines in the vector field visualization.
        Only applicable for "line" and "arrow" types.
    visibility : tuple, optional
        A tuple of three boolean values indicating the visibility of the slices
        in the x, y, and z dimensions, respectively.

    Returns
    -------
    VectorField
        A vector field object.
    """

    if cross_section is None:
        cross_section = np.asarray(field.shape[:3], dtype=np.int32) // 2

    obj = VectorField(
        field,
        actor_type=actor_type,
        cross_section=cross_section,
        colors=colors,
        scales=scales,
        opacity=opacity,
        thickness=thickness,
        visibility=visibility,
    )
    return obj


@register_wgpu_render_function(VectorField, VectorFieldThinLineMaterial)
def register_vector_field_thin_shaders(wobject):
    """Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldThinShader(wobject)
    return compute_shader, render_shader


@register_wgpu_render_function(VectorField, VectorFieldLineMaterial)
def register_vector_field_shaders(wobject):
    """Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldShader(wobject)
    return compute_shader, render_shader


@register_wgpu_render_function(VectorField, VectorFieldArrowMaterial)
def register_vector_field_arrow_shaders(wobject):
    """Register PeaksActor shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = VectorFieldComputeShader(wobject)
    render_shader = VectorFieldArrowShader(wobject)
    return compute_shader, render_shader


def surface(
    vertices,
    faces,
    *,
    material="phong",
    colors=None,
    texture=None,
    texture_axis="xy",
    opacity=1.0,
):
    """Create a surface mesh actor from vertices and faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        The vertex positions of the surface mesh.
    faces : ndarray, shape (M, 3)
        The indices of the vertices that form each triangular face.
    material : str, optional
        The material type for the surface mesh. Options are 'phong' and 'basic'. This
        option only works with colors is passed.
    colors : ndarray, shape (N, 3) or (N, 4) or tuple (3,) or tuple (4,), optional
        RGB or RGBA values in the range [0, 1].
    texture : str, optional
        Path to the texture image file.
    texture_axis : str, optional
        The axis to generate UV coordinates for the texture. Options are 'xy', 'yz',
        and 'xz'. This option only works with texture is passed.
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

        if isinstance(colors, np.ndarray) and colors.shape[0] == vertices.shape[0]:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
                colors=colors,
            )
            mat = _create_mesh_material(
                material=material, mode="vertex", opacity=opacity
            )
        elif isinstance(colors, (tuple, list)) and len(colors) == 3:
            geo = buffer_to_geometry(
                positions=vertices.astype("float32"),
                indices=faces.astype("int32"),
            )
            mat = _create_mesh_material(color=colors, opacity=opacity)
        else:
            raise ValueError(
                "Colors must be either an ndarray with shape (N, 3) or (N, 4), "
                "or a tuple/list of length 3 for RGB colors."
            )
    elif texture is not None:
        if not os.path.exists(texture):
            raise FileNotFoundError(f"Texture file '{texture}' not found.")

        logging.warning(
            "texture option currently only supports planar projection,"
            " the plane can be provided by texture_axis parameter."
        )

        tex = load_image_texture(texture)
        texcoords = generate_planar_uvs(vertices, axis=texture_axis)
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"),
            indices=faces.astype("int32"),
            texcoords=texcoords.astype("float32"),
        )
        mat = MeshBasicMaterial(map=tex, opacity=opacity)
    else:
        geo = buffer_to_geometry(
            positions=vertices.astype("float32"), indices=faces.astype("int32")
        )
        mat = _create_mesh_material(material=material, opacity=opacity)

    obj = create_mesh(geo, mat)
    return obj


class SphGlyph(Mesh):
    """Visualize a spherical harmonic glyph with different features.

    Parameters
    ----------
    coeffs : ndarray, shape (X, Y, Z, N)
        The spherical harmonics coefficients. X, Y, Z denotes the position and N
        represents the number of coefficients.
    sphere : tuple
            Vertices and faces of the sphere to use for the glyph.
    basis_type : str, optional
        The type of basis to use for the spherical harmonics.
        Options are 'standard', 'descoteaux07'.
    color_type : str, optional
        The type of color mapping to use for the spherical glyph.
        Options are 'sign' and 'orientation'.
    shininess : float, optional
        The shininess of the material for the spherical glyph.
    """

    def __init__(
        self,
        coeffs,
        sphere,
        *,
        basis_type="standard",
        color_type="sign",
        shininess=50,
    ):
        """Visualize a spherical harmonic glyph with different features.

        Parameters
        ----------
        coeffs : ndarray, shape (X, Y, Z, N)
            The spherical harmonics coefficients. X, Y, Z denotes the position and N
            represents the number of coefficients.
        sphere : tuple
            Vertices and faces of the sphere to use for the glyph.
        basis_type : str, optional
            The type of basis to use for the spherical harmonics.
            Options are 'standard', 'descoteaux07'.
        color_type : str, optional
            The type of color mapping to use for the spherical glyph.
            Options are 'sign' and 'orientation'.
        shininess : float, optional
            The shininess of the material for the spherical glyph.
        """

        if not isinstance(coeffs, np.ndarray):
            raise TypeError("The attribute 'coeffs' must be a numpy ndarray.")
        elif coeffs.ndim != 4:
            raise ValueError(
                (
                    "The attribute 'coeffs' must be a 4D numpy ndarray "
                    "with shape (X, Y, Z, N)."
                )
            )
        elif coeffs.shape[-1] < 1:
            raise ValueError(
                "The last dimension of 'coeffs' must be greater than 0, "
                f"but got {coeffs.shape[-1]}"
            )

        if not isinstance(sphere, tuple):
            raise TypeError(
                "The attribute 'sphere' must be a tuple containing vertices and faces."
            )
        elif (
            len(sphere) != 2
            or not isinstance(sphere[0], np.ndarray)
            or not isinstance(sphere[1], np.ndarray)
        ):
            raise TypeError(
                "The attribute 'sphere' must be a tuple containing two numpy ndarrays "
                "(vertices, faces)."
            )

        self.n_coeff = coeffs.shape[-1]
        self.data_shape = coeffs.shape[:3]
        l_max = get_lmax(self.n_coeff, basis_type=basis_type)
        self.color_type = 0 if color_type == "sign" else 1

        vertices, faces = sphere[0], sphere[1]
        positions = np.tile(vertices, (np.prod(self.data_shape), 1)).astype(np.float32)
        positions[0] = np.asarray(self.data_shape)
        self.scaled_vertices = np.zeros_like(positions, dtype=np.float32)

        self.vertices_per_glyph = vertices.shape[0]
        self.faces_per_glyph = faces.shape[0]

        self.indices = faces.reshape(-1).astype(np.int32)
        indices = np.tile(faces, (np.prod(self.data_shape), 1)).astype(np.int32)

        self.radii = np.zeros((self.vertices_per_glyph,), dtype=np.float32)

        for i in range(0, indices.shape[0], faces.shape[0]):
            start = i
            end = start + faces.shape[0]
            indices[start:end] += (i // faces.shape[0]) * self.vertices_per_glyph

        geo = buffer_to_geometry(
            positions=positions.astype("float32"),
            indices=indices.astype("int32"),
            colors=np.ones_like(positions, dtype="float32"),
            normals=np.zeros_like(positions).astype("float32"),
        )

        mat = SphGlyphMaterial(
            l_max=l_max,
            color_mode="vertex",
            flat_shading=False,
            shininess=shininess,
            specular="#494949",
            side="front",
        )

        B_mat = create_sh_basis_matrix(vertices, l_max)
        self.sh_coeff = coeffs.reshape(-1).astype("float32")
        self.sf_func = B_mat.reshape(-1).astype("float32")
        self.sphere = vertices.astype("float32")

        super().__init__(geometry=geo, material=mat)


def sph_glyph(
    coeffs, *, sphere=None, basis_type="standard", color_type="sign", shininess=50
):
    """Visualize a spherical harmonic glyph with different features.

    Parameters
    ----------
    coeffs : ndarray, shape (X, Y, Z, N)
        The spherical harmonics coefficients. X, Y, Z denotes the position and N
        represents the number of coefficients.
    sphere : {str, tuple}, optional
        The name of the sphere to use or a tuple containing the phi and theta
        segments for a custom sphere.
        Available options for the named spheres:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'
    basis_type : str, optional
        The type of basis to use for the spherical harmonics.
        Options are 'standard', 'descoteaux07'.
    color_type : str, optional
        The type of color mapping to use for the spherical glyph.
        Options are 'sign' and 'orientation'.
    shininess : float, optional
        The shininess of the material for the spherical glyph.

    Returns
    -------
    SphGlyph
        A spherical glyph object.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError("The attribute 'coeffs' must be a numpy ndarray.")
    elif coeffs.ndim != 4:
        raise ValueError(
            "The attribute 'coeffs' must be a 4D numpy ndarray with shape (X, Y, Z, N)."
        )

    if sphere is None:
        sphere = "symmetric362"

    if isinstance(sphere, str):
        sphere = fp.prim_sphere(name=sphere)
    elif (
        isinstance(sphere, tuple)
        and isinstance(sphere[0], int)
        and isinstance(sphere[1], int)
    ):
        sphere = fp.prim_sphere(gen_faces=True, phi=sphere[0], theta=sphere[1])
    else:
        raise TypeError(
            "The attribute 'sphere' must be a string or tuple containing two integers."
        )

    if not isinstance(basis_type, str):
        raise TypeError("The attribute 'basis_type' must be a string.")
    elif basis_type not in ["standard", "descoteaux07"]:
        raise ValueError(
            "The attribute 'basis_type' must be either 'standard' or 'descoteaux07'."
        )

    if not isinstance(color_type, str):
        raise TypeError("The attribute 'color_type' must be a string.")
    if color_type not in ["sign", "orientation"]:
        raise ValueError(
            "The attribute 'color_type' must be either 'sign' or 'orientation'."
        )

    if not isinstance(shininess, (int, float)):
        raise TypeError("The attribute 'shininess' must be an integer or float.")

    obj = SphGlyph(
        coeffs=coeffs,
        sphere=sphere,
        basis_type=basis_type,
        color_type=color_type,
        shininess=shininess,
    )
    return obj


@register_wgpu_render_function(SphGlyph, SphGlyphMaterial)
def register_glyph_shaders(wobject):
    """Register Glyph shaders.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to register shaders for.

    Returns
    -------
    tuple
        A tuple containing the compute shader and the render shader.
    """
    compute_shader = SphGlyphComputeShader(wobject)
    render_shader = MeshPhongShader(wobject)
    return compute_shader, render_shader
