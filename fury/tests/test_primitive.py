import numpy as np
import numpy.testing as npt
import fury.primitive as fp


def test_vertices_primitives():
    l_primitives = [(fp.prim_square, (4, 3)),
                    (fp.prim_box, (8, 3))]

    for func, shape in l_primitives:
        vertices, _ = func()

        npt.assert_equal(vertices.shape, shape)
        npt.assert_equal(np.mean(vertices), 0)
        npt.assert_equal(vertices.min(), -.5)
        npt.assert_equal(vertices.max(), 0.5)


def test_triangles_primitives():
    l_primitives = [(fp.prim_square, (2, 3)),
                    (fp.prim_box, (12, 3))]

    for func, shape in l_primitives:
        vertices, triangles = func()
        npt.assert_equal(triangles.shape, shape)
        npt.assert_equal(list(set(np.concatenate(triangles, axis=None))),
                         list(range(len(vertices))))


def test_spheres_primitives():
    l_primitives = [('symmetric362', 362, 720), ('symmetric642', 642, 1280),
                    ('symmetric724', 724, 1444), ('repulsion724', 724, 1444),
                    ('repulsion100', 100, 196), ('repulsion200', 200, 396)]

    for name, nb_verts, nb_triangles in l_primitives:
        verts, faces = fp.prim_sphere(name)
        npt.assert_equal(verts.shape, (nb_verts, 3))
        npt.assert_almost_equal(np.mean(verts), 0)
        npt.assert_equal(len(faces), nb_triangles)
        npt.assert_equal(list(set(np.concatenate(faces, axis=None))),
                         list(range(len(verts))))

    npt.assert_raises(ValueError, fp.prim_sphere, 'sym362')
