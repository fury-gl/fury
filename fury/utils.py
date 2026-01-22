"""Utility functions for 3D graphics and visualization.

This module contains various utility functions for 3D graphics and
visualization, including trilinear interpolation, affine transformations,
normal calculations, and grid generation. These functions are designed
to work with numpy arrays and are useful for manipulating 3D data
structures, such as meshes and point clouds.
"""

import logging

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    find_objects,
    generate_binary_structure,
    label,
    map_coordinates,
)
from scipy.special import factorial, lpmv

from fury.transform import cart2sphere

_FACE_QUAD_OFFSETS = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # axis 0 (YZ plane)
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # axis 1 (XZ plane)
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # axis 2 (XY plane)
    ],
    dtype=np.int8,
)

_FACE_DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int8,
)

_FACE_AXES = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
_FACE_SIGNS = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)


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


def generate_planar_uvs(vertices, *, axis="xy"):
    """Generate UVs by projecting vertices onto a plane.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Array of vertex coordinates in 3D space.
    axis : str, optional
        The plane onto which to project the vertices. Options are 'xy', 'xz', or 'yz'.

    Returns
    -------
    ndarray
        Array of UV coordinates, shape (N, 2), where N is the number of vertices.
    """

    if axis not in ("xy", "xz", "yz"):
        raise ValueError("axis must be one of 'xy', 'xz', or 'yz'.")

    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 2:
        raise ValueError("vertices must be a 2D array with shape (N, 3) with N > 2.")

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords

    if (range_coords[0] == 0 or range_coords[1] == 0) and axis == "xy":
        raise ValueError("Cannot generate UVs for flat geometry in the XY plane.")
    if (range_coords[0] == 0 or range_coords[2] == 0) and axis == "xz":
        raise ValueError("Cannot generate UVs for flat geometry in the XZ plane.")
    if (range_coords[1] == 0 or range_coords[2] == 0) and axis == "yz":
        raise ValueError("Cannot generate UVs for flat geometry in the YZ plane.")

    uvs = np.zeros((len(vertices), 2))
    for i, v in enumerate(vertices):
        if axis == "xy":
            uvs[i] = [
                (v[0] - min_coords[0]) / range_coords[0],
                (v[1] - min_coords[1]) / range_coords[1],
            ]
        elif axis == "xz":
            uvs[i] = [
                (v[0] - min_coords[0]) / range_coords[0],
                (v[2] - min_coords[2]) / range_coords[2],
            ]
        elif axis == "yz":
            uvs[i] = [
                (v[1] - min_coords[1]) / range_coords[1],
                (v[2] - min_coords[2]) / range_coords[2],
            ]
    return uvs


def create_sh_basis_matrix(vertices, l_max):
    """Create a SH basis matrix for real spherical harmonics.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Array of vertex coordinates in 3D space, where N is the number of vertices.
    l_max : int
        Maximum spherical harmonic degree.

    Returns
    -------
    ndarray, shape (N, (l_max + 1) ** 2)
        Matrix of spherical harmonic basis functions evaluated at the vertices.
    """

    if (
        not isinstance(vertices, np.ndarray)
        or vertices.ndim != 2
        or vertices.shape[1] != 3
    ):
        raise ValueError("vertices must be a 2D array with shape (N, 3).")
    if not isinstance(l_max, int) or l_max < 0:
        raise ValueError("l_max must be a non-negative integer.")

    _, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    phi[phi < 0] += 2 * np.pi
    n_vertices = vertices.shape[0]
    cos_theta = np.cos(theta)
    n_coeffs = (l_max + 1) ** 2
    B = np.zeros((n_vertices, n_coeffs), dtype=np.float32)

    col_idx = 0

    for order in range(l_max + 1):
        for m in range(-order, order + 1):
            norm = np.sqrt(
                ((2 * order + 1) / (4 * np.pi))
                * (factorial(order - abs(m)) / factorial(order + abs(m)))
            )

            legendre_poly = lpmv(abs(m), order, cos_theta)

            if m > 0:
                sh_values = np.sqrt(2) * norm * legendre_poly * np.cos(m * phi)
            elif m < 0:
                sh_values = np.sqrt(2) * norm * legendre_poly * np.sin(abs(m) * phi)
            else:  # m == 0
                sh_values = norm * legendre_poly

            B[:, col_idx] = sh_values
            col_idx += 1

    return B


def get_lmax(n_coeffs, *, basis_type="standard"):
    """Get the maximum degree (l_max) from the number of coefficients.

    Parameters
    ----------
    n_coeffs : int
        The number of spherical harmonic coefficients.
    basis_type : str, optional
        The type of spherical harmonic basis.
        Can be "standard" or "descoteaux07". If None, defaults to "standard".

    Returns
    -------
    int
        The maximum spherical harmonic degree (l_max).
    """

    if not isinstance(n_coeffs, int) or n_coeffs < 1:
        raise ValueError("n_coeffs must be a non-zero, positive integer.")

    if basis_type not in ("standard", "descoteaux07"):
        raise ValueError("basis_type must be one of 'standard' or 'descoteaux07'.")

    if basis_type == "standard":
        return int(np.rint(np.sqrt(n_coeffs + 1) - 1))
    elif basis_type == "descoteaux07":
        return int(np.rint(np.sqrt(2 * n_coeffs - 0.5) - 1.5))


def get_n_coeffs(l_max, *, basis_type="standard"):
    """Get the number of spherical harmonic coefficients from the maximum degree.

    Parameters
    ----------
    l_max : int
        The maximum spherical harmonic degree.
    basis_type : str, optional
        The type of spherical harmonic basis.
        Can be "standard" or "descoteaux07". If None, defaults to "standard".

    Returns
    -------
    int
        The number of spherical harmonic coefficients.
    """

    if not isinstance(l_max, int) or l_max < 0:
        raise ValueError("l_max must be a non-negative integer.")

    if basis_type not in ("standard", "descoteaux07"):
        raise ValueError("basis_type must be one of 'standard' or 'descoteaux07'.")

    if basis_type == "standard":
        return (l_max + 1) ** 2
    elif basis_type == "descoteaux07":
        return int((l_max + 1.5) ** 2 + 0.5) // 2


def get_transformed_cube_bounds(affine_matrix, vertex1, vertex2):
    """Get the min and max ranges of a transformed cube.

    Parameters
    ----------
    affine_matrix : ndarray, shape (4, 4)
        The affine transformation matrix to apply to the cube vertices.
    vertex1 : ndarray, shape (3,)
        The coordinates of one corner of the cube.
    vertex2 : ndarray, shape (3,)
        The coordinates of the opposite corner of the cube.

    Returns
    -------
    list
        A list containing the min and max ranges of the transformed cube in the format
        [[min_x, min_y, min_z], [max_x, max_y, max_z]].
    """

    if len(vertex1) != 3 or len(vertex2) != 3:
        raise ValueError("vertex1 and vertex2 must be 3D coordinates.")
    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        raise ValueError("affine_matrix must be a 4x4 numpy array.")

    x1, y1, z1 = vertex1
    x2, y2, z2 = vertex2

    vertices = np.array(
        [
            [x1, y1, z1, 1],
            [x2, y1, z1, 1],
            [x1, y2, z1, 1],
            [x1, y1, z2, 1],
            [x2, y2, z1, 1],
            [x2, y1, z2, 1],
            [x1, y2, z2, 1],
            [x2, y2, z2, 1],
        ]
    )

    transformed_vertices = vertices @ affine_matrix.T

    transformed_vertices = transformed_vertices[:, :3]

    min_vals = np.min(transformed_vertices, axis=0)
    max_vals = np.max(transformed_vertices, axis=0)

    return [min_vals, max_vals]


def extract_surface_voxels(volume, label_value, *, structuring_element=None):
    """Extract boundary voxel coordinates for a label within a volume.

    Parameters
    ----------
    volume : ndarray
        Labeled volume containing the objects of interest; can be a full volume
        or a cropped sub-section.
    label_value : int
        The label identifying the object whose surface voxels should be returned.
    structuring_element : ndarray, optional
        Structuring element defining the desired connectivity. If None,
        a default 1-connectivity structuring element is used.

    Returns
    -------
    tuple or None
        Returns (surface_coords, object_mask) when a surface is found, where
        surface_coords has shape (N, 3) ordered as (x, y, z). Returns None when
        the label does not have exposed voxels.
    """

    if structuring_element is None:
        structuring_element = generate_binary_structure(rank=3, connectivity=1)

    object_mask = volume == label_value

    volume_interior_mask = binary_erosion(
        object_mask, structure=structuring_element, border_value=0
    )
    background_and_surface_mask = np.logical_not(volume_interior_mask)
    surface_mask = np.logical_and(object_mask, background_and_surface_mask)

    x_coords, y_coords, z_coords = np.nonzero(surface_mask)

    if x_coords.size == 0:
        logging.info(f"No surface found for label {label_value}.")
        return None

    surface_coords = np.stack((x_coords, y_coords, z_coords), axis=1)
    return surface_coords, object_mask


def face_generation(coords, axis, sign):
    """Generate voxel face corners for scalar or vector inputs.

    Parameters
    ----------
    coords : array_like
        Voxel origins shaped as (..., 3). Arrays are broadcast with `axis` and
        `sign` to produce multiple quads simultaneously.
    axis : array_like
        Axis orthogonal to each face (0=x, 1=y, 2=z).
    sign : array_like
        Orientation of each face normal (+1 or -1).

    Returns
    -------
    ndarray
        Array of quad vertices with shape (..., 4, 3), where the leading
        dimensions match the broadcasted inputs.
    """
    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3).")

    axis = np.broadcast_to(np.asarray(axis), (coords.shape[0],)).astype(
        np.int8, copy=False
    )
    sign = np.broadcast_to(np.asarray(sign), (coords.shape[0],)).astype(
        np.int8, copy=False
    )

    flat_quads = np.empty((coords.shape[0], 4, 3), dtype=coords.dtype)

    for axis_id in range(3):
        axis_mask = axis == axis_id
        if not np.any(axis_mask):
            continue

        coords_subset = coords[axis_mask].copy()
        subset_signs = sign[axis_mask]

        positive_mask = subset_signs > 0
        coords_subset[positive_mask, axis_id] += 1

        offsets = _FACE_QUAD_OFFSETS[axis_id].astype(coords_subset.dtype, copy=False)
        quad_values = coords_subset[:, None, :] + offsets

        negative_mask = subset_signs < 0
        if np.any(negative_mask):
            quad_values[negative_mask] = quad_values[negative_mask][:, ::-1, :]

        flat_quads[axis_mask] = quad_values

    return flat_quads


def voxel_mesh_by_object(
    volume, *, connectivity=1, spacing=(1.0, 1.0, 1.0), triangulate=False
):
    """Build a watertight mesh from a 3D volume where objects are volume > 0.

    Parameters
    ----------
    volume : ndarray
        3D array where objects are represented by non-zero values.
    connectivity : int, optional
        Possible options are {1, 2, 3}. 1 -> 6-neighborhood, 2 -> 18, 3 -> 26.
    spacing : tuple, optional
        Voxel spacing along each axis (dx, dy, dz).
    triangulate : bool, optional
        Whether to triangulate the resulting mesh faces.

    Returns
    -------
    dict
        A dictionary where keys are object labels and values are dictionaries
        with 'verts' and 'faces' of the generated meshes.
    """

    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError("volume must be a 3D numpy array.")

    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be one of {1, 2, 3}.")

    if spacing is None or len(spacing) != 3:
        raise ValueError("spacing must be a tuple of 3 elements.")

    if triangulate not in (True, False):
        raise ValueError("triangulate must be a boolean value.")

    struct = generate_binary_structure(rank=3, connectivity=connectivity)
    labels, n = label(volume > 0, structure=struct)

    if n == 0:
        logging.info("No objects found in the volume.")
        return {}

    boxes = find_objects(labels)
    spacing = np.asarray(spacing)

    objects = {}

    for lbl in range(1, n + 1):
        slc = boxes[lbl - 1]
        if slc is None:
            continue
        sublab = labels[slc]
        surface_data = extract_surface_voxels(sublab, lbl, structuring_element=struct)

        if surface_data is None:
            continue

        surface_coords, object_mask = surface_data
        if surface_coords.size == 0:
            continue

        offset = np.array([slc[0].start, slc[1].start, slc[2].start])
        global_surface = surface_coords + offset

        neighbors = surface_coords[:, None, :] + _FACE_DIRECTIONS
        nx = neighbors[..., 0]
        ny = neighbors[..., 1]
        nz = neighbors[..., 2]

        X, Y, Z = object_mask.shape
        valid = (nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y) & (nz >= 0) & (nz < Z)

        occupied = np.zeros_like(valid, dtype=bool)
        valid_idx = np.where(valid)
        occupied[valid_idx] = object_mask[nx[valid_idx], ny[valid_idx], nz[valid_idx]]
        exposed = np.logical_not(occupied)

        if not np.any(exposed):
            continue

        surface_expanded = np.broadcast_to(global_surface[:, None, :], neighbors.shape)
        face_coords = surface_expanded[exposed]
        face_axes = np.broadcast_to(
            _FACE_AXES, (surface_coords.shape[0], _FACE_AXES.size)
        )[exposed]
        face_signs = np.broadcast_to(
            _FACE_SIGNS, (surface_coords.shape[0], _FACE_SIGNS.size)
        )[exposed]

        quads = face_generation(face_coords, face_axes, face_signs)
        quad_vertices = quads.reshape(-1, 3)

        unique_vertices, inverse = np.unique(quad_vertices, axis=0, return_inverse=True)
        faces_array = inverse.reshape(-1, 4)

        if triangulate:
            faces_array = np.concatenate(
                (faces_array[:, [0, 1, 2]], faces_array[:, [0, 2, 3]]), axis=0
            )

        verts_array = (unique_vertices * spacing).astype(np.float32, copy=False)
        faces_array = faces_array.astype(np.int32, copy=False)

        objects[lbl] = {"verts": verts_array, "faces": faces_array}

    if not objects:
        logging.info("No objects were extracted from the volume.")

    return objects
