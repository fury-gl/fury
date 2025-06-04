"""Utility functions for 3D graphics and visualization.

This module contains various utility functions for 3D graphics and
visualization, including trilinear interpolation, affine transformations,
normal calculations, and grid generation. These functions are designed
to work with numpy arrays and are useful for manipulating 3D data
structures, such as meshes and point clouds.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from fury.lib import AffineTransform, Group, RecursiveTransform, WorldObject
from fury.material import validate_opacity


def map_coordinates_3d_4d(input_array, indices):
    """Evaluate input_array at the given indices using trilinear interpolation.

    Uses trilinear interpolation to sample values from a 3D or 4D array at
    specified coordinates.

    Parameters
    ----------
    input_array : ndarray
        3D or 4D array to be sampled.
    indices : ndarray
        Coordinates at which to evaluate `input_array`. Should have shape
        (N, D) where N is the number of points and D is the dimensionality
        of the input array.

    Returns
    -------
    ndarray
        1D or 2D array of values evaluated at `indices`. Shape will be
        (N,) for 3D input or (N, M) for 4D input, where M is the size of
        the last dimension of `input_array`.

    Raises
    ------
    ValueError
        If input_array is not 3D or 4D.
    """
    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i], indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def apply_affine(aff, pts):
    """Apply affine matrix `aff` to points `pts`.

    Returns result of application of `aff` to the *right* of `pts`. The
    coordinate dimension of `pts` should be the last.

    Parameters
    ----------
    aff : ndarray, shape (N, N)
        Homogeneous affine matrix. For 3D points, will be 4 by 4. The affine
        will be applied on the left of `pts`.
    pts : ndarray, shape (..., N-1)
        Points to transform, where the last dimension contains the coordinates
        of each point. For 3D, the last dimension will be length 3.

    Returns
    -------
    ndarray, shape (..., N-1)
        Transformed points with same shape as input `pts`.

    Notes
    -----
    Copied from nibabel to remove dependency.

    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3. The transformation is computed as::

        res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
        transformed_pts = res.T

    Examples
    --------
    >>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    >>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    >>> # Just to show that in the simple 3D case, it is equivalent to:
    >>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)
    >>> # But `pts` can be a more complicated shape:
    >>> pts = pts.reshape((2,2,3))
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[[14, 14, 24],
            [16, 17, 28]],
    <BLANKLINE>
           [[20, 23, 36],
            [24, 29, 44]]]...)
    """
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1, :-1]
    trans = aff[:-1, -1]
    res = np.dot(pts, rzs.T) + trans[None, :]
    return res.reshape(shape)


def asbytes(s):
    """Convert string to bytes using latin1 encoding.

    Parameters
    ----------
    s : str or bytes
        Input string or bytes object.

    Returns
    -------
    bytes
        Input encoded as bytes using latin1 encoding.
    """
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def get_grid_cells_position(shapes, *, aspect_ratio=16 / 9.0, dim=None):
    """Construct a XY-grid based on the cells content shape.

    Generate coordinates for a grid of cells with specified content shapes.
    The grid follows a row-major order with the top left corner at (0,0,0).

    Parameters
    ----------
    shapes : list of tuple of int
        The shape (width, height) of every cell content.
    aspect_ratio : float, optional
        Aspect ratio of the grid (width/height). Default is 16/9.
    dim : tuple of int, optional
        Dimension (nb_rows, nb_cols) of the grid. If None, dimensions are
        calculated automatically to match the aspect_ratio.

    Returns
    -------
    ndarray, shape (N, 3)
        3D coordinates of every grid cell center, where N is the number of cells.

    Raises
    ------
    ValueError
        If the provided `dim` is too small to contain all elements.

    Notes
    -----
    The grid width and height are determined by the largest width and height
    among all cell contents. X coordinates increase from left to right while
    Y coordinates decrease from top to bottom.
    """
    cell_shape = np.r_[np.max(shapes, axis=0), 0]
    cell_aspect_ratio = cell_shape[0] / cell_shape[1]

    count = len(shapes)
    if dim is None:
        # Compute the number of rows and columns.
        n_cols = np.ceil(np.sqrt(count * aspect_ratio / cell_aspect_ratio))
        n_rows = np.ceil(count / n_cols)
    else:
        n_rows, n_cols = dim

    if n_cols * n_rows < count:
        msg = "Size is too small, it cannot contain at least {} elements."
        raise ValueError(msg.format(count))

    # Use indexing="xy" so the cells are in row-major (C-order). Also,
    # the Y coordinates are negative so the cells are order from top to bottom.
    X, Y, Z = np.meshgrid(np.arange(n_cols), -np.arange(n_rows), [0], indexing="xy")
    return cell_shape * np.array([X.flatten(), Y.flatten(), Z.flatten()]).T


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors in-place.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        Array of vectors to normalize.

    Returns
    -------
    ndarray, shape (N, 3)
        The input array, normalized in-place.

    Notes
    -----
    Vectors are normalized by dividing each component by the vector length.
    Zero-length vectors will cause division by zero.
    """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def normals_from_v_f(vertices, faces):
    """Calculate vertex normals from vertices and faces.

    Compute surface normals for each vertex based on the faces that include it.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Array of vertex coordinates.
    faces : ndarray, shape (M, 3)
        Array of triangle indices, where each element is an index into vertices.

    Returns
    -------
    ndarray, shape (N, 3)
        Array of calculated normals, one per vertex.

    Notes
    -----
    The normal for each face is calculated and then accumulated for each
    vertex. The final normals are normalized to unit length.
    """
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


def tangents_from_direction_of_anisotropy(normals, direction):
    """Calculate tangents from normals and a direction of anisotropy.

    Parameters
    ----------
    normals : ndarray, shape (N, 3)
        Array of surface normals per vertex.
    direction : array_like, shape (3,)
        Vector representing the direction of anisotropy.

    Returns
    -------
    ndarray, shape (N, 3)
        Array of calculated tangents, one per vertex.

    Notes
    -----
    The tangent vectors are calculated by first finding the binormal
    (cross product of normal and tangent direction), then taking the
    cross product of the normal and binormal.
    """
    tangents = np.cross(normals, direction)
    binormals = normalize_v3(np.cross(normals, tangents))
    return normalize_v3(np.cross(normals, binormals))


def triangle_order(vertices, face):
    """Determine the winding order of a given triangle face.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Array of vertices making up a shape.
    face : ndarray, shape (3,)
        Array of 3 indices representing a single triangle face.

    Returns
    -------
    bool
        True if the order is counter-clockwise (CCW), False otherwise.

    Notes
    -----
    The winding order is determined using the sign of the determinant
    of a 4x4 matrix containing the triangle vertices.
    """
    v1 = vertices[face[0]]
    v2 = vertices[face[1]]
    v3 = vertices[face[2]]

    # https://stackoverflow.com/questions/40454789/computing-face-normals-and-winding
    m_orient = np.ones((4, 4))
    m_orient[0, :3] = v1
    m_orient[1, :3] = v2
    m_orient[2, :3] = v3
    m_orient[3, :3] = 0

    val = np.linalg.det(m_orient)

    return bool(val > 0)


def change_vertices_order(triangle):
    """Change the vertices order of a given triangle.

    Parameters
    ----------
    triangle : ndarray, shape (3,)
        Array of 3 vertex indices making up a triangle.

    Returns
    -------
    ndarray, shape (3,)
        New array of vertex indices in the opposite winding order.

    Notes
    -----
    Reverses the winding order by swapping the first and last vertices.
    """
    return np.array([triangle[2], triangle[1], triangle[0]])


def fix_winding_order(vertices, triangles, *, clockwise=False):
    """Return triangles with a fixed winding order.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Array of vertices corresponding to a shape.
    triangles : ndarray, shape (M, 3)
        Array of triangles corresponding to a shape.
    clockwise : bool, optional
        Desired triangle order type: True for clockwise, False for
        counter-clockwise (default).

    Returns
    -------
    ndarray, shape (M, 3)
        The triangles with corrected winding order.

    Notes
    -----
    Clockwise means that the three vertices, in order, rotate clockwise around
    the triangle's center. This function ensures all triangles have the
    specified winding order by checking each triangle and correcting as needed.
    """
    corrected_triangles = triangles.copy()
    desired_order = clockwise
    for nb, face in enumerate(triangles):
        current_order = triangle_order(vertices, face)
        if desired_order != current_order:
            corrected_triangles[nb] = change_vertices_order(face)
    return corrected_triangles


def set_group_visibility(group, visibility):
    """Set the visibility of a group of actors.

    Parameters
    ----------
    group : Group
        The group of actors to set visibility for.
    visibility : tuple or list of bool
        If a single boolean value is provided, it sets the visibility for all
        actors in the group. If a tuple or list is provided, it sets the
        visibility for each actor in the group individually.
    """
    if not isinstance(group, Group):
        raise TypeError("group must be an instance of Group.")

    if not isinstance(visibility, (tuple, list)):
        group.visible = visibility
        return

    for idx, actor in enumerate(group.children):
        actor.visible = visibility[idx]


def set_group_opacity(group, opacity):
    """Set the opacity of the group of actors.

    Parameters
    ----------
    group : Group
        The group of actors to set opacity for.
    opacity : float
        The opacity value to set for the group of actors,
        ranging from 0 (fully transparent) to 1 (opaque).
    """
    if not isinstance(group, Group):
        raise TypeError("group must be an instance of Group.")

    opacity = validate_opacity(opacity)

    for child in group.children:
        child.material.opacity = opacity


def validate_slices_group(group):
    """Validate the slices in a group.

    Parameters
    ----------
    group : Group
        The group of actors to validate.

    Raises
    ------
    TypeError
        If the group is not an instance of Group.
    ValueError
        If the group does not contain exactly 3 children.
    AttributeError
        If the children do not have the required material plane attribute.
    """
    if not isinstance(group, Group):
        raise TypeError("group must be an instance of Group.")

    if len(group.children) != 3:
        raise ValueError(
            f"Group must contain exactly 3 children. {len(group.children)}"
        )

    if not hasattr(group.children[0].material, "plane"):
        raise AttributeError(
            "Children do not have the required material plane attribute for slices."
        )


def get_slices(group):
    """Get the current positions of the slices.

    Parameters
    ----------
    group : Group
        The group of actors to get the slices from.

    Returns
    -------
    ndarray
        An array containing the current positions of the slices.
    """
    validate_slices_group(group)
    return np.asarray([child.material.plane[-1] for child in group.children])


def show_slices(group, position):
    """Show the slices at the specified position.

    Parameters
    ----------
    group : Group
        The group of actors to get the slices from.
    position : tuple
        A tuple containing the positions of the slices in the 3D space.
    """
    validate_slices_group(group)

    for i, child in enumerate(group.children):
        a, b, c, _ = child.material.plane
        child.material.plane = (a, b, c, position[i])


def apply_affine_to_group(group, affine):
    """Apply a transformation to all actors in a group.

    Parameters
    ----------
    group : Group
        The group of actors to apply the transformation to.
    affine : ndarray, shape (4, 4)
        The transformation to apply to the actors in the group.
    """
    if not isinstance(group, Group):
        raise TypeError("group must be an instance of Group.")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 numpy array.")

    for child in group.children:
        apply_affine_to_actor(child, affine)


def apply_affine_to_actor(actor, affine):
    """Apply a transformation to an actor.

    Parameters
    ----------
    actor : WorldObject
        The actor to apply the transformation to.
    affine : ndarray, shape (4, 4)
        The transformation to apply to the actor.
    """
    if not isinstance(actor, WorldObject):
        raise TypeError("actor must be an instance of WorldObject.")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 numpy array.")

    affine_transform = AffineTransform(
        state_basis="matrix", matrix=affine, is_camera_space=True
    )
    recursive_transform = RecursiveTransform(affine_transform)
    actor.local = affine_transform
    actor.world = recursive_transform
