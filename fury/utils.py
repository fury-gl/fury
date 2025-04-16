import numpy as np
from scipy.ndimage import map_coordinates

from fury.lib import Group
from fury.material import validate_opacity


def map_coordinates_3d_4d(input_array, indices):
    """Evaluate input_array at the given indices using trilinear interpolation.

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

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

    Returns result of application of `aff` to the *right* of `pts`.  The
    coordinate dimension of `pts` should be the last.
    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3 - maybe it will just be N by 3. The return value is the
    transformed points, in this case::
    res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
    transformed_pts = res.T
    This routine is more general than 3D, in that `aff` can have any shape
    (N,N), and `pts` can have any shape, as long as the last dimension is for
    the coordinates, and is therefore length N-1.

    Parameters
    ----------
    aff : (N, N) array-like

        Homogeneous affine, for 3D points, will be 4 by 4. Contrary to first
        appearance, the affine will be applied on the left of `pts`.
    pts : (..., N-1) array-like
        Points, where the last dimension contains the coordinates of each
        point.  For 3D, the last dimension will be length 3.

    Returns
    -------
    transformed_pts : (..., N-1) array
        transformed points

    Notes
    -----
    Copied from nibabel to remove dependency.

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
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def get_grid_cells_position(shapes, *, aspect_ratio=16 / 9.0, dim=None):
    """Construct a XY-grid based on the cells content shape.

    This function generates the coordinates of every grid cell. The width and
    height of every cell correspond to the largest width and the largest height
    respectively. The grid dimensions will automatically be adjusted to respect
    the given aspect ratio unless they are explicitly specified.

    The grid follows a row-major order with the top left corner being at
    coordinates (0,0,0) and the bottom right corner being at coordinates
    (nb_cols*cell_width, -nb_rows*cell_height, 0). Note that the X increases
    while the Y decreases.

    Parameters
    ----------
    shapes : list of tuple of int
        The shape (width, height) of every cell content.
    aspect_ratio : float (optional)
        Aspect ratio of the grid (width/height). Default: 16:9.
    dim : tuple of int (optional)
        Dimension (nb_rows, nb_cols) of the grid, if provided.

    Returns
    -------
    ndarray
        3D coordinates of every grid cell.

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
    """Normalize a numpy array of 3 component vectors shape=(N, 3).

    Parameters
    ----------
    array : ndarray
        Shape (N, 3)

    Returns
    -------
    norm_array

    """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def normals_from_v_f(vertices, faces):
    """Calculate normals from vertices and faces.

    Parameters
    ----------
    verices : ndarray
    faces : ndarray

    Returns
    -------
    normals : ndarray
        Shape same as vertices

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
    """Calculate tangents from normals and a 3D vector representing the
       direction of anisotropy.

    Parameters
    ----------
    normals : normals, represented as 2D ndarrays (Nx3) (one per vertex)
    direction : tuple (3,) or array (3,)

    Returns
    -------
    output : array (N, 3)
        Tangents, represented as 2D ndarrays (Nx3).

    """
    tangents = np.cross(normals, direction)
    binormals = normalize_v3(np.cross(normals, tangents))
    return normalize_v3(np.cross(normals, binormals))


def triangle_order(vertices, faces):
    """Determine the winding order of a given set of vertices and a triangle.

    Parameters
    ----------
    vertices : ndarray
        array of vertices making up a shape
    faces : ndarray
        array of triangles

    Returns
    -------
    order : int
        If the order is counter clockwise (CCW), returns True.
        Otherwise, returns False.

    """
    v1 = vertices[faces[0]]
    v2 = vertices[faces[1]]
    v3 = vertices[faces[2]]

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
    triangle : ndarray, shape(1, 3)
        array of 3 vertices making up a triangle

    Returns
    -------
    new_triangle : ndarray, shape(1, 3)
        new array of vertices making up a triangle in the opposite winding
        order of the given parameter

    """
    return np.array([triangle[2], triangle[1], triangle[0]])


def fix_winding_order(vertices, triangles, *, clockwise=False):
    """Return corrected triangles.

    Given an ordering of the triangle's three vertices, a triangle can appear
    to have a clockwise winding or counter-clockwise winding.
    Clockwise means that the three vertices, in order, rotate clockwise around
    the triangle's center.

    Parameters
    ----------
    vertices : ndarray
        array of vertices corresponding to a shape
    triangles : ndarray
        array of triangles corresponding to a shape
    clockwise : bool
        triangle order type: clockwise (default) or counter-clockwise.

    Returns
    -------
    corrected_triangles : ndarray
        The corrected order of the vert parameter

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


def _valid_slices(group):
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
    position : ndarray
        An array containing the current positions of the slices.
    """
    _valid_slices(group)
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
    _valid_slices(group)

    for i, child in enumerate(group.children):
        a, b, c, _ = child.material.plane
        child.material.plane = (a, b, c, position[i])
