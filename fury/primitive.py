from os.path import join as pjoin
import numpy as np
from fury.data import DATA_DIR


SPHERE_FILES = {
    'symmetric362': pjoin(DATA_DIR, 'evenly_distributed_sphere_362.npz'),
    'symmetric642': pjoin(DATA_DIR, 'evenly_distributed_sphere_642.npz'),
    'symmetric724': pjoin(DATA_DIR, 'evenly_distributed_sphere_724.npz'),
    'repulsion724': pjoin(DATA_DIR, 'repulsion724.npz'),
    'repulsion100': pjoin(DATA_DIR, 'repulsion100.npz'),
    'repulsion200': pjoin(DATA_DIR, 'repulsion200.npz')
}


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


def prim_sphere(name='symmetric362'):
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

    return res['vertices'], res['faces']
