"""Module dedicated for basic primitive."""
from os.path import join as pjoin
import numpy as np
from fury.data import DATA_DIR
from fury.transform import cart2sphere, euler_matrix
from fury.utils import fix_winding_order
from scipy.spatial import ConvexHull
from scipy.spatial import transform
import math


SPHERE_FILES = {
    'symmetric362': pjoin(DATA_DIR, 'evenly_distributed_sphere_362.npz'),
    'symmetric642': pjoin(DATA_DIR, 'evenly_distributed_sphere_642.npz'),
    'symmetric724': pjoin(DATA_DIR, 'evenly_distributed_sphere_724.npz'),
    'repulsion724': pjoin(DATA_DIR, 'repulsion724.npz'),
    'repulsion100': pjoin(DATA_DIR, 'repulsion100.npz'),
    'repulsion200': pjoin(DATA_DIR, 'repulsion200.npz')
}


def faces_from_sphere_vertices(vertices):
    """
    Triangulate a set of vertices on the sphere.

    Parameters
    ----------
    vertices : (M, 3) ndarray
        XYZ coordinates of vertices on the sphere.

    Returns
    -------
    faces : (N, 3) ndarray
        Indices into vertices; forms triangular faces.

    """
    hull = ConvexHull(vertices, qhull_options='Qbb Qc')
    faces = np.ascontiguousarray(hull.simplices)
    if len(vertices) < 2**16:
        return np.asarray(faces, np.uint16)
    else:
        return faces


def repeat_primitive_function(func, centers, func_args=[],
                              directions=(1, 0, 0), colors=(255, 0, 0),
                              scale=1):
    """Repeat Vertices and triangles of a specific primitive function.

    It could be seen as a glyph. The primitive function should generate and
    return vertices and faces

    Parameters
    ----------
    func : callable
        primitive functions
    centers : ndarray, shape (N, 3)
        Superquadrics positions
    func_args : args
        primitive functions arguments/parameters
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the cone.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scale : ndarray, shape (N) or (N,3) or float or int, optional
        The height of the cone.

    Returns
    -------
    big_vertices: ndarray
        Expanded vertices at the centers positions
    big_triangles: ndarray
        Expanded triangles that composed our shape to duplicate
    big_colors : ndarray
        Expanded colors applied to all vertices/faces

    """
    # Get faces
    _, faces = func()
    if len(func_args) == 1:
        func_args = np.squeeze(np.array([func_args] * centers.shape[0]))
    elif len(func_args) != centers.shape[0]:
        raise IOError("sq_params should 1 or equal to the numbers \
                        of centers")

    vertices = np.concatenate([func(i)[0] for i in func_args])
    return repeat_primitive(vertices=vertices, faces=faces, centers=centers,
                            directions=directions, colors=colors, scale=scale,
                            have_tiled_verts=True)


def repeat_primitive(vertices, faces, centers, directions=(1, 0, 0),
                     colors=(255, 0, 0), scale=1, have_tiled_verts=False):
    """Repeat Vertices and triangles of a specific primitive shape.

    It could be seen as a glyph.

    Parameters
    ----------
    vertices: ndarray
        vertices coords to duplicate at the centers positions
    triangles: ndarray
        triangles that composed our shape to duplicate
    centers : ndarray, shape (N, 3)
        Superquadrics positions
    directions : ndarray, shape (N, 3) or tuple (3,), optional
        The orientation vector of the cone.
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scale : ndarray, shape (N) or (N,3) or float or int, optional
        The height of the cone.
    have_tiled_verts : bool
        option to control if we need to duplicate vertices of a shape or not

    Returns
    -------
    big_vertices: ndarray
        Expanded vertices at the centers positions
    big_triangles: ndarray
        Expanded triangles that composed our shape to duplicate
    big_colors : ndarray
        Expanded colors applied to all vertices/faces
    big_centers : ndarray
        Expanded centers for all vertices/faces

    """
    # duplicated vertices if needed
    if not have_tiled_verts:
        vertices = np.tile(vertices, (centers.shape[0], 1))
    big_vertices = vertices
    # Get unit shape
    unit_verts_size = vertices.shape[0] // centers.shape[0]
    unit_triangles_size = faces.shape[0]

    # scale them
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale = np.repeat(scale, unit_verts_size, axis=0)
        scale = scale.reshape((big_vertices.shape[0], 1))
    big_vertices *= scale

    # update triangles
    big_triangles = np.array(np.tile(faces,
                                     (centers.shape[0], 1)),
                             dtype=np.int32)
    big_triangles += np.repeat(np.arange(0,
                                         centers.shape[0] * unit_verts_size,
                                         step=unit_verts_size),
                               unit_triangles_size,
                               axis=0).reshape((big_triangles.shape[0],
                                                1))

    def normalize_input(arr, arr_name=''):
        if isinstance(arr, (tuple, list, np.ndarray)) and len(arr) == 3 and \
                not all(isinstance(i, (list, tuple, np.ndarray)) for i in arr):
            return np.array([arr] * centers.shape[0])
        elif isinstance(arr, np.ndarray) and len(arr) == 1:
            return np.repeat(arr, centers.shape[0], axis=0)
        elif len(arr) != len(centers):
            msg = "{} size should be 1 or ".format(arr_name)
            msg += "equal to the numbers of centers"
            raise IOError(msg)
        else:
            return np.array(arr)

    # update colors
    colors = normalize_input(colors, 'colors')
    big_colors = np.repeat(colors, unit_verts_size, axis=0)

    # update orientations
    directions = normalize_input(directions, 'directions')
    for pts, dirs in enumerate(directions):
        ai, aj, ak = transform.Rotation.from_rotvec(np.pi / 2 * dirs). \
            as_euler('zyx')
        rotation_matrix = euler_matrix(ai, aj, ak)
        big_vertices[pts * unit_verts_size: (pts + 1) * unit_verts_size] = \
            np.dot(rotation_matrix[:3, :3],
                   big_vertices[pts * unit_verts_size:
                                (pts + 1) * unit_verts_size].T).T

    # apply centers position
    big_centers = np.repeat(centers, unit_verts_size, axis=0)
    big_vertices += big_centers

    return big_vertices, big_triangles, big_colors, big_centers


def prim_square():
    """Return vertices and triangles for a square geometry.

    Returns
    -------
    vertices: ndarray
        4 vertices coords that composed our square
    triangles: ndarray
        2 triangles that composed our square

    """
    vertices = np.array([[-.5, -.5, 0.0],
                         [-.5, 0.5, 0.0],
                         [0.5, 0.5, 0.0],
                         [0.5, -.5, 0.0]])
    triangles = np.array([[0, 1, 2],
                          [2, 3, 0]], dtype='i8')
    return vertices, triangles


def prim_box():
    """Return vertices and triangle for a box geometry.

    Returns
    -------
    vertices: ndarray
        8 vertices coords that composed our box
    triangles: ndarray
        12 triangles that composed our box

    """
    vertices = np.array([[-.5, -.5, -.5],
                         [-.5, -.5, 0.5],
                         [-.5, 0.5, -.5],
                         [-.5, 0.5, 0.5],
                         [0.5, -.5, -.5],
                         [0.5, -.5, 0.5],
                         [0.5, 0.5, -.5],
                         [0.5, 0.5, 0.5]])
    triangles = np.array([[0, 6, 4],
                          [0, 2, 6],
                          [0, 3, 2],
                          [0, 1, 3],
                          [2, 7, 6],
                          [2, 3, 7],
                          [4, 6, 7],
                          [4, 7, 5],
                          [0, 4, 5],
                          [0, 5, 1],
                          [1, 5, 7],
                          [1, 7, 3]], dtype='i8')
    return vertices, triangles


def prim_sphere(name='symmetric362', gen_faces=False):
    """Provide vertices and triangles of the spheres.

    Parameters
    ----------
    name : str
        which sphere - one of:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'
    gen_faces : bool, optional
        If True, triangulate a set of vertices on the sphere to get the faces.
        Otherwise, we load the saved faces from a file. Default: False

    Returns
    -------
    vertices: ndarray
        vertices coords that composed our sphere
    triangles: ndarray
        triangles that composed our sphere

    Examples
    --------
    >>> import numpy as np
    >>> from fury.primitive import prim_sphere
    >>> verts, faces = prim_sphere('symmetric362')
    >>> verts.shape == (362, 3)
    True
    >>> faces.shape == (720, 3)
    True

    """
    fname = SPHERE_FILES.get(name)
    if fname is None:
        raise ValueError('No sphere called "%s"' % name)
    res = np.load(fname)

    verts = res['vertices'].copy()
    faces = faces_from_sphere_vertices(verts) if gen_faces else res['faces']
    faces = fix_winding_order(res['vertices'], faces)
    return res['vertices'], faces


def prim_superquadric(roundness=(1, 1), sphere_name='symmetric362'):
    """Provide vertices and triangles of a superquadrics.

    Parameters
    ----------
    roundness : tuple, optional
        parameters (Phi and Theta) that control the shape of the superquadric

    sphere_name : str, optional
        which sphere - one of:
        * 'symmetric362'
        * 'symmetric642'
        * 'symmetric724'
        * 'repulsion724'
        * 'repulsion100'
        * 'repulsion200'

    Returns
    -------
    vertices: ndarray
        vertices coords that composed our sphere
    triangles: ndarray
        triangles that composed our sphere

    Examples
    --------
    >>> import numpy as np
    >>> from fury.primitive import prim_superquadric
    >>> verts, faces = prim_superquadric(roundness=(1, 1))
    >>> verts.shape == (362, 3)
    True
    >>> faces.shape == (720, 3)
    True

    """
    def _fexp(x, p):
        """Return a different kind of exponentiation."""
        return np.sign(x) * (np.abs(x) ** p)

    sphere_verts, sphere_triangles = prim_sphere(sphere_name)
    _, sphere_phi, sphere_theta = cart2sphere(*sphere_verts.T)

    phi, theta = roundness
    x = _fexp(np.sin(sphere_phi), phi) * _fexp(np.cos(sphere_theta), theta)
    y = _fexp(np.sin(sphere_phi), phi) * _fexp(np.sin(sphere_theta), theta)
    z = _fexp(np.cos(sphere_phi), phi)
    xyz = np.vstack([x, y, z]).T

    vertices = np.ascontiguousarray(xyz)

    return vertices, sphere_triangles


def prim_tetrahedron():
    """Return vertices and triangles for a tetrahedron.

    This shape has a side length of two units.

    Returns
    -------
    pyramid_vert: numpy.ndarray
        4 vertices coordinates
    triangles: numpy.ndarray
        4 triangles representing the tetrahedron

    """

    pyramid_vert = np.array([[0.5, 0.5, 0.5],
                             [0.5, -0.5, -0.5],
                             [-0.5, 0.5, -0.5],
                             [-0.5, -0.5, 0.5]])

    pyramid_triag = np.array([[2, 0, 1],
                              [0, 3, 2],
                              [0, 3, 1],
                              [1, 2, 3]], dtype='i8')

    return pyramid_vert, pyramid_triag


def prim_icosahedron():
    """Return vertices and triangles for icosahedron.

    Returns
    -------
    icosahedron_vertices: numpy.ndarray
        12 vertices coordinates to the icosahedron
    icosahedron_mesh: numpy.ndarray
        20 triangles representing the tetrahedron

    """
    phi = (1 + math.sqrt(5)) / 2.0

    icosahedron_vertices = np.array([[-1.0, 0.0, phi],
                                     [0.0, phi, 1.0],
                                     [1.0, 0.0, phi],
                                     [-phi, 1.0, 0.0],
                                     [0.0, phi, -1.0],
                                     [phi, 1.0, 0.0],
                                     [-phi, -1.0, 0.0],
                                     [0.0, -phi, 1.0],
                                     [phi, -1.0, 0.0],
                                     [-1.0, 0.0, -phi],
                                     [0.0, -phi, -1.0],
                                     [1.0, 0.0, -phi]])

    icosahedron_mesh = np.array([[1, 0, 2],
                                 [2, 5, 1],
                                 [5, 4, 1],
                                 [3, 1, 4],
                                 [0, 1, 3],
                                 [0, 6, 3],
                                 [9, 3, 6],
                                 [8, 2, 7],
                                 [2, 0, 7],
                                 [0, 7, 6],
                                 [5, 2, 8],
                                 [11, 5, 8],
                                 [11, 4, 5],
                                 [9, 11, 4],
                                 [4, 3, 9],
                                 [11, 10, 8],
                                 [8, 10, 7],
                                 [6, 7, 10],
                                 [10, 9, 6],
                                 [9, 10, 11]], dtype='i8')

    return icosahedron_vertices, icosahedron_mesh
