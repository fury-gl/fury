import numpy as np
import numpy.testing as npt
import fury.primitive as fp

def test_vertices_primitives():
    l_primitives = [(fp.square, (4, 3)),
                    (fp.box, (8, 3))]

    for func, shape in l_primitives:
        vertices, _ = func()

        npt.assert_equal(vertices.shape, shape)
        npt.assert_equal(np.mean(vertices), 0)
        npt.assert_equal(vertices.min(), -.5)
        npt.assert_equal(vertices.max(), 0.5)


def test_triangles_primitives():
    l_primitives = [(fp.square, (2, 3)),
                    (fp.box, (12, 3))]

    for func, shape in l_primitives:
        vertices, triangles = func()
        npt.assert_equal(triangles.shape, shape)
        npt.assert_equal(list(set(np.concatenate(triangles, axis=None))),
                         list(range(len(vertices))))
