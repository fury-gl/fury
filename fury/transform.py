"""Transformation functions for 3D graphics."""

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot  # type: ignore

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}


def euler_matrix(ai, aj, ak, *, axes="sxyz"):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First Euler angle in radians.
    aj : float
        Second Euler angle in radians.
    ak : float
        Third Euler angle in radians.
    axes : str or tuple, optional
        One of 24 axis sequences as string or encoded tuple. Default is 'sxyz'.

    Returns
    -------
    ndarray (4, 4)
        Homogeneous rotation matrix.

    Notes
    -----
    Code modified from the work of Christoph Gohlke:
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    Examples
    --------
    >>> import numpy
    >>> R = euler_matrix(1, 2, 3, axes='syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, axes=(0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes=axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes=axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def sphere2cart(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : array_like
        Radius.
    theta : array_like
        Inclination or polar angle.
    phi : array_like
        Azimuth angle.

    Returns
    -------
    x : array
        X coordinate(s) in Cartesian space.
    y : array
        Y coordinate(s) in Cartesian space.
    z : array
        Z coordinate(s) in Cartesian space.

    Notes
    -----
    Imagine a sphere with center (0,0,0). Orient it with the z axis
    running south-north, the y axis running west-east and the x axis
    from posterior to anterior. `theta` (the inclination angle) is the
    angle to rotate from the z-axis (the zenith) around the y-axis,
    towards the x axis. Thus the rotation is counter-clockwise from the
    point of view of positive y. `phi` (azimuth) gives the angle of
    rotation around the z-axis towards the y axis. The rotation is
    counter-clockwise from the point of view of positive z.

    Equivalently, given a point P on the sphere, with coordinates x, y,
    z, `theta` is the angle between P and the z-axis, and `phi` is
    the angle between the projection of P onto the XY plane, and the X
    axis.

    Geographical nomenclature designates theta as 'co-latitude', and phi
    as 'longitude'.

    See these pages for more details:
    * http://en.wikipedia.org/wiki/Spherical_coordinate_system
    * http://mathworld.wolfram.com/SphericalCoordinates.html

    We have deliberately named this function ``sphere2cart`` rather than
    ``sph2cart`` to distinguish it from the Matlab function of that
    name, which uses a different convention.
    """
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = r * np.cos(theta)
    x, y, z = np.broadcast_arrays(x, y, z)
    return x, y, z


def cart2sphere(x, y, z):
    r"""Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : array_like
        X coordinate in Cartesian space.
    y : array_like
        Y coordinate in Cartesian space.
    z : array_like
        Z coordinate in Cartesian space.

    Returns
    -------
    r : array
        Radius.
    theta : array
        Inclination (polar) angle in range [0, π].
    phi : array
        Azimuth angle in range [-π, π].

    Notes
    -----
    Uses the same convention as sphere2cart. The inclination angle theta
    is in range [0, π] and the azimuth angle phi is in range [-π, π].

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    See sphere2cart for detailed description of the coordinate convention.
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.0)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi


def translate(translation):
    """Create a transformation matrix for translation.

    Parameters
    ----------
    translation : ndarray (3,)
        Translation vector in x, y and z directions.

    Returns
    -------
    ndarray (4, 4)
        Homogeneous transformation matrix with translation parameters in the
        last column.

    Examples
    --------
    >>> import numpy as np; import fury
    >>> tran = np.array([0.3, 0.2, 0.25])
    >>> transform = fury.transform.translate(tran)
    >>> transform
    array([[1.  , 0.  , 0.  , 0.3 ],
           [0.  , 1.  , 0.  , 0.2 ],
           [0.  , 0.  , 1.  , 0.25],
           [0.  , 0.  , 0.  , 1.  ]])
    """
    iden = np.identity(4)
    translation = np.append(translation, 0).reshape(-1, 1)

    t = np.array(
        [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
        np.float32,
    )
    translation = np.multiply(t, translation)
    translation = np.add(iden, translation)

    return translation


def rotate(quat):
    """Create a transformation matrix for rotation using quaternion.

    Parameters
    ----------
    quat : ndarray (4,)
        Rotation quaternion in form [x, y, z, w].

    Returns
    -------
    ndarray (4, 4)
        Homogeneous transformation matrix for rotation.

    Examples
    --------
    >>> import numpy as np; import fury
    >>> quat = np.array([0.259, 0.0, 0.0, 0.966])
    >>> rotation = fury.transform.rotate(quat)
    >>> rotation
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.86586979, -0.50026944,  0.        ],
           [ 0.        ,  0.50026944,  0.86586979,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    iden = np.identity(3)
    rotation_mat = Rot.from_quat(quat).as_matrix()

    iden = np.append(iden, [[0, 0, 0]]).reshape(-1, 3)

    rotation_mat = np.dot(iden, rotation_mat)
    iden = np.array([[0, 0, 0, 1]]).reshape(-1, 1)

    rotation_mat = np.concatenate((rotation_mat, iden), axis=1)
    return rotation_mat


def scale(scales):
    """Create a transformation matrix for scaling.

    Parameters
    ----------
    scales : ndarray (3,)
        Scale factors for x, y and z directions.

    Returns
    -------
    ndarray (4, 4)
        Homogeneous transformation matrix with scale factors along the diagonal.

    Examples
    --------
    >>> import numpy as np; import fury
    >>> scales = np.array([2.0, 1.0, 0.5])
    >>> transform = fury.transform.scale(scales)
    >>> transform
    array([[2. , 0. , 0. , 0. ],
           [0. , 1. , 0. , 0. ],
           [0. , 0. , 0.5, 0. ],
           [0. , 0. , 0. , 1. ]])
    """
    scale_mat = np.identity(4)
    scales = np.append(scales, [1])

    for i in range(len(scales)):
        scale_mat[i][i] = scales[i]

    return scale_mat


def apply_transformation(vertices, transformation):
    """Apply transformation matrix to vertices.

    Parameters
    ----------
    vertices : ndarray (n, 3)
        Array of vertices to be transformed.
    transformation : ndarray (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    ndarray (n, 3)
        Transformed vertices.

    Notes
    -----
    This function multiplies the transformation matrix with the vertices to
    transform them in 3D space. The vertices are converted to homogeneous
    coordinates before multiplication.
    """
    shape = vertices.shape
    temp = np.full((shape[0], 1), 1)
    vertices = np.concatenate((vertices, temp), axis=1)

    vertices = np.dot(transformation, vertices.T)
    vertices = vertices.T
    vertices = vertices[:, : shape[1]]

    return vertices


def transform_from_matrix(matrix):
    """Decompose transformation matrix into components.

    Parameters
    ----------
    matrix : ndarray (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    translation : ndarray (3,)
        Translation vector (tx, ty, tz) extracted from the matrix.
    rotation : ndarray (4,)
        Rotation parameters as [angle, rx, ry, rz] where angle is in degrees
        and rx, ry, rz represent the rotation vector direction.
    scale : ndarray (3,)
        Scale factors (sx, sy, sz) extracted from the matrix.

    Notes
    -----
    The function extracts the translation directly from the last column of
    the matrix, calculates the scale by computing the norm of each of the
    first three columns, and then derives the rotation matrix after
    normalizing for scale.
    """
    translate = matrix[:, -1:].reshape((-1,))[:-1]

    temp = matrix[:, :3][:3]
    sx = np.linalg.norm(temp[:, :1])
    sy = np.linalg.norm(temp[:, 1:-1])
    sz = np.linalg.norm(temp[:, -1:])
    scale = np.array([sx, sy, sz])

    rot_matrix = temp / scale[None, :]
    rotation = Rot.from_matrix(rot_matrix)
    rot_vec = rotation.as_rotvec()
    angle = np.linalg.norm(rot_vec)
    rotation = [np.rad2deg(angle), *rot_vec]

    return translate, rotation, scale
