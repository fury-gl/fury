import math

import numpy as np
import numpy.testing as npt

import fury.primitive as fp


def test_vertices_primitives():
    # Tests the default vertices of all the built in primitive shapes.
    l_primitives = [
        (fp.prim_square, (4, 3), -0.5, 0.5, 0, {}),
        (fp.prim_triangle, (3, 3), -0.5, 0.5, -0.0555555555, {}),
        (fp.prim_box, (24, 3), -0.5, 0.5, 0, {"detailed": True}),
        (fp.prim_box, (8, 3), -0.5, 0.5, 0, {"detailed": False}),
        (fp.prim_tetrahedron, (4, 3), -0.5, 0.5, 0, {}),
        (fp.prim_star, (10, 3), -0.47552825814757677, 1 / 2, 0, {}),
        (fp.prim_rhombicuboctahedron, (24, 3), -0.5, 0.5, 0, {}),
        (fp.prim_frustum, (8, 3), -0.5, 0.5, 0, {}),
    ]

    for func, shape, e_min, e_max, e_mean, kwargs in l_primitives:
        vertices, _ = func(**kwargs)
        npt.assert_equal(vertices.shape, shape)
        npt.assert_almost_equal(np.mean(vertices), e_mean)
        npt.assert_equal(vertices.min(), e_min)
        npt.assert_equal(vertices.max(), e_max)

    vertices, _ = fp.prim_star(dim=3)
    npt.assert_equal(vertices.shape, (12, 3))
    npt.assert_almost_equal(np.mean(vertices), 0, decimal=7)
    # Ensure no coordinate exceeds the specified outer radius
    npt.assert_(vertices.min() >= -1 / 2)
    npt.assert_equal(vertices.max(), 1 / 2)


def test_vertices_primitives_icosahedron():
    vertices, _ = fp.prim_icosahedron()
    shape = (12, 3)
    phi = (1 + math.sqrt(5)) / 2.0
    npt.assert_equal(vertices.shape, shape)
    npt.assert_equal(np.mean(vertices), 0)
    npt.assert_equal(vertices.min(), -phi)
    npt.assert_equal(vertices.max(), phi)


def test_vertices_primitives_octagonalprism():
    # Testing the default vertices of the primitive octagonal prism.
    vertices, _ = fp.prim_octagonalprism()
    shape = (16, 3)
    two = (1 + float(f"{math.sqrt(2):.7f}")) / 4

    npt.assert_equal(vertices.shape, shape)
    npt.assert_equal(np.mean(vertices), 0)
    npt.assert_equal(vertices.min(), -two)
    npt.assert_equal(vertices.max(), two)


def test_vertices_primitives_pentagonalprism():
    # Testing the default vertices of the primitive pentagonal prism.
    vertices, _ = fp.prim_pentagonalprism()
    lower_face = vertices[:, 0:2][0:5,]
    upper_face = vertices[:, 0:2][5:10,]
    centroid_upper = np.mean(upper_face, 0)
    centroid_lower = np.mean(lower_face, 0)
    shape = (10, 3)
    npt.assert_equal(vertices.shape, shape)
    # This test will check whether the z-axis vertex dispersion is correct
    npt.assert_almost_equal(np.mean(vertices[:, 2]), 0)
    # check if the centroid of the upper face is at the origin
    npt.assert_almost_equal(centroid_upper, np.array([0, 0]))
    # check if the centroid of the lower face is at the origin
    npt.assert_almost_equal(centroid_lower, np.array([0, 0]))


def test_vertices_primitives_triangularprism():
    # Testing the default vertices of the primitive triangular prism.
    vertices, _ = fp.prim_triangularprism()
    shape = (6, 3)
    three = float(f"{math.sqrt(3):.7f}")
    npt.assert_equal(vertices.shape, shape)
    npt.assert_equal(np.mean(vertices), 0)
    npt.assert_equal(vertices.min(), -1 / three)
    npt.assert_equal(vertices.max(), 1 / 2)


def test_triangles_primitives():
    l_primitives = [
        (fp.prim_square, (2, 3)),
        (fp.prim_triangle, (1, 3)),
        (fp.prim_box, (12, 3)),
        (fp.prim_tetrahedron, (4, 3)),
        (fp.prim_icosahedron, (20, 3)),
    ]

    for func, shape in l_primitives:
        vertices, triangles = func()
        npt.assert_equal(triangles.shape, shape)
        npt.assert_equal(
            list(set(np.concatenate(triangles, axis=None))), list(range(len(vertices)))
        )


def test_spheres_primitives():
    l_primitives = [
        ("symmetric362", 362, 720),
        ("symmetric642", 642, 1280),
        ("symmetric724", 724, 1444),
        ("repulsion724", 724, 1444),
        ("repulsion100", 100, 196),
        ("repulsion200", 200, 396),
    ]

    for name, nb_verts, nb_triangles in l_primitives:
        verts, faces = fp.prim_sphere(name=name)
        npt.assert_equal(verts.shape, (nb_verts, 3))
        npt.assert_almost_equal(np.mean(verts), 0)
        npt.assert_equal(len(faces), nb_triangles)
        npt.assert_equal(
            list(set(np.concatenate(faces, axis=None))), list(range(len(verts)))
        )

    npt.assert_raises(
        ValueError,
        fp.prim_sphere,
        name="sym362",
        gen_faces=False,
        phi=None,
        theta=None,
    )

    l_primitives = [
        (10, 10, 82, 160),
        (20, 20, 362, 720),
        (10, 12, 102, 200),
        (22, 20, 398, 792),
    ]

    for nb_phi, nb_theta, nb_verts, nb_triangles in l_primitives:
        verts, faces = fp.prim_sphere(phi=nb_phi, theta=nb_theta)
        npt.assert_equal(verts.shape, (nb_verts, 3))
        npt.assert_almost_equal(np.mean(verts), 0)
        npt.assert_equal(len(faces), nb_triangles)
        npt.assert_equal(
            list(set(np.concatenate(faces, axis=None))), list(range(len(verts)))
        )


def test_superquadric_primitives():
    # test default, should be like a sphere 362
    sq_verts, sq_faces = fp.prim_superquadric()
    s_verts, s_faces = fp.prim_sphere(name="symmetric362")

    npt.assert_equal(sq_verts.shape, s_verts.shape)
    npt.assert_equal(sq_faces.shape, s_faces.shape)

    npt.assert_almost_equal(sq_verts, s_verts)

    # Apply roundness
    sq_verts, sq_faces = fp.prim_superquadric(roundness=(2, 3))
    npt.assert_equal(sq_verts.shape, s_verts.shape)
    npt.assert_equal(sq_faces.shape, s_faces.shape)

    # TODO: We need to check some superquadrics shape


def test_cylinder_primitive():
    verts, faces = fp.prim_cylinder(radius=0.5, height=1, sectors=10)
    npt.assert_equal(verts.shape, (44, 3))
    npt.assert_almost_equal(np.mean(verts), 0, decimal=1)
    npt.assert_equal(verts.min(), -0.5)
    npt.assert_equal(verts.max(), 0.5)

    # basic tests for triangle
    npt.assert_equal(faces.shape, (40, 3))
    npt.assert_equal(
        np.unique(np.concatenate(faces, axis=None)).tolist(), list(range(len(verts)))
    )

    verts, faces = fp.prim_cylinder(radius=0.5, height=1, sectors=10, capped=False)
    npt.assert_equal(verts.shape, (22, 3))
    npt.assert_almost_equal(np.mean(verts), 0, decimal=1)
    npt.assert_equal(verts.min(), -0.5)
    npt.assert_equal(verts.max(), 0.5)
    npt.assert_equal(
        np.unique(np.concatenate(faces, axis=None)).tolist(), list(range(len(verts)))
    )


def test_arrow_primitive():
    verts, faces = fp.prim_arrow(
        height=1.0, resolution=10, tip_length=0.35, tip_radius=0.1, shaft_radius=0.03
    )
    npt.assert_equal(verts.shape, (36, 3))
    npt.assert_almost_equal(np.mean(verts), 0, decimal=1)
    # x-axis
    npt.assert_equal(verts.T[0].min(), 0)
    npt.assert_equal(verts.T[0].max(), 1)
    # y and z axes
    npt.assert_equal(verts.T[1:2].min(), -0.1)
    npt.assert_equal(verts.T[1:2].max(), 0.1)
    npt.assert_equal(np.mean(verts[1, 2].T), 0.0)

    # basic tests for triangle
    npt.assert_equal(faces.shape, (50, 3))
    npt.assert_equal(
        np.unique(np.concatenate(faces, axis=None)).tolist(), list(range(len(verts)))
    )


def test_cone_primitive():
    verts, faces = fp.prim_cone()
    npt.assert_equal(verts.shape, (12, 3))
    npt.assert_equal(verts.min(), -0.5)
    npt.assert_equal(verts.max(), 0.5)
    npt.assert_almost_equal(np.mean(verts), 0, decimal=1)

    # tests for triangles
    npt.assert_equal(faces.shape, (20, 3))
    npt.assert_equal(
        np.unique(np.concatenate(faces, axis=None)).tolist(), list(range(len(verts)))
    )

    # test warnings
    npt.assert_raises(ValueError, fp.prim_cone, radius=0.5, height=1, sectors=2)


def test_repeat_primitive():
    # init variables
    verts, faces = fp.prim_square()
    centers = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])

    for i in [-1, 1]:
        dirs = dirs * i
        res = fp.repeat_primitive(verts, faces, centers, directions=dirs, colors=colors)

        big_verts, big_faces, big_colors, big_centers = res

        npt.assert_equal(big_verts.shape[0], verts.shape[0] * centers.shape[0])
        npt.assert_equal(big_faces.shape[0], faces.shape[0] * centers.shape[0])
        npt.assert_equal(big_colors.shape[0], verts.shape[0] * centers.shape[0])
        npt.assert_equal(big_centers.shape[0], verts.shape[0] * centers.shape[0])

        npt.assert_equal(
            np.unique(np.concatenate(big_faces, axis=None)).tolist(),
            list(range(len(big_verts))),
        )

        # translate the squares primitives centers to be the origin
        big_vert_origin = big_verts - np.repeat(centers, 4, axis=0)

        # three repeated primitives
        sq1, sq2, sq3 = big_vert_origin.reshape([3, 12])

        #  primitives directed toward different directions must not be the same
        npt.assert_equal(np.any(np.not_equal(sq1, sq2)), True)
        npt.assert_equal(np.any(np.not_equal(sq1, sq3)), True)
        npt.assert_equal(np.any(np.not_equal(sq2, sq3)), True)

        npt.assert_equal(big_vert_origin.min(), -0.5)
        npt.assert_equal(big_vert_origin.max(), 0.5)
        npt.assert_equal(np.mean(big_vert_origin), 0)

        npt.assert_equal(big_colors.min(), 0)
        npt.assert_equal(big_colors.max(), 1)


def test_repeat_primitive_function():
    # init variables
    centers = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255
    phi_theta = np.array([[1, 1], [1, 2], [2, 1]])

    _ = fp.repeat_primitive_function(
        fp.prim_superquadric,
        centers,
        func_args=phi_theta,
        directions=dirs,
        colors=colors,
    )

    # big_verts, big_faces, big_colors, big_centers = res

    # npt.assert_equal(big_verts.shape[0],  verts.shape[0] * centers.shape[0])


def test_disk_primitive():
    verts, faces = fp.prim_disk()
    npt.assert_equal(verts.shape, (37, 3))
    npt.assert_equal(faces.shape, (36, 3))
    npt.assert_almost_equal(np.mean(verts), 0, decimal=2)

    npt.assert_array_equal(verts[0], [0, 0, 0])

    # Check outer points are at radius distance from center
    outer_verts = verts[1:]
    distances = np.sqrt(outer_verts[:, 0] ** 2 + outer_verts[:, 1] ** 2)
    npt.assert_almost_equal(distances, 0.5, decimal=6)

    # Check all z-coordinates are 0 (flat disk)
    npt.assert_array_equal(verts[:, 2], np.zeros(len(verts)))

    # Check with custom parameters
    radius = 1.5
    sectors = 20
    verts, faces = fp.prim_disk(radius=radius, sectors=sectors)
    npt.assert_equal(verts.shape, (sectors + 1, 3))
    npt.assert_equal(faces.shape, (sectors, 3))

    # Check outer points are at custom radius
    outer_verts = verts[1:]
    distances = np.sqrt(outer_verts[:, 0] ** 2 + outer_verts[:, 1] ** 2)
    npt.assert_almost_equal(distances, radius, decimal=6)

    # Test triangle indices reference all vertices
    npt.assert_equal(
        np.unique(np.concatenate(faces, axis=None)).tolist(), list(range(len(verts)))
    )

    npt.assert_raises(TypeError, fp.prim_disk, radius=1.0, sectors=10.5)
    npt.assert_raises(ValueError, fp.prim_disk, radius=1.0, sectors=7)


def test_prim_ring_vertices_and_triangles():
    verts, faces = fp.prim_ring()

    # Expected shapes:
    # (circumferential_segments * (radial_segments + 1), 3) for vertices
    # (circumferential_segments * radial_segments * 2, 3) for faces
    expected_verts_shape_default = (32 * (1 + 1), 3)
    expected_faces_shape_default = (32 * 1 * 2, 3)

    npt.assert_equal(verts.shape, expected_verts_shape_default)
    npt.assert_equal(faces.shape, expected_faces_shape_default)

    npt.assert_almost_equal(verts.min(), -1.0, decimal=7)
    npt.assert_almost_equal(verts.max(), 1.0, decimal=7)

    # Check radial distances for inner and outer vertices
    inner_radius_default = 0.5
    outer_radius_default = 1.0
    radial_segments_default = 1

    # Check inner ring vertices
    inner_ring_verts_indices = np.arange(0, verts.shape[0], radial_segments_default + 1)
    inner_ring_verts = verts[inner_ring_verts_indices, :2]
    distances_inner = np.sqrt(inner_ring_verts[:, 0] ** 2 + inner_ring_verts[:, 1] ** 2)
    npt.assert_almost_equal(distances_inner, inner_radius_default, decimal=6)

    # Check outer ring vertices
    outer_ring_verts_indices = np.arange(
        radial_segments_default, verts.shape[0], radial_segments_default + 1
    )
    outer_ring_verts = verts[outer_ring_verts_indices, :2]
    distances_outer = np.sqrt(outer_ring_verts[:, 0] ** 2 + outer_ring_verts[:, 1] ** 2)
    npt.assert_almost_equal(distances_outer, outer_radius_default, decimal=6)

    # Check that all vertices are referenced by triangles
    npt.assert_equal(
        list(set(np.concatenate(faces, axis=None))), list(range(len(verts)))
    )

    custom_inner_radius_1 = 0.1
    custom_outer_radius_1 = 2.0
    custom_radial_segments_1 = 3
    custom_circumferential_segments_1 = 16

    verts_c1, faces_c1 = fp.prim_ring(
        inner_radius=custom_inner_radius_1,
        outer_radius=custom_outer_radius_1,
        radial_segments=custom_radial_segments_1,
        circumferential_segments=custom_circumferential_segments_1,
    )

    expected_verts_shape_c1 = (
        custom_circumferential_segments_1 * (custom_radial_segments_1 + 1),
        3,
    )
    expected_faces_shape_c1 = (
        custom_circumferential_segments_1 * custom_radial_segments_1 * 2,
        3,
    )

    npt.assert_equal(verts_c1.shape, expected_verts_shape_c1)
    npt.assert_equal(faces_c1.shape, expected_faces_shape_c1)
    npt.assert_almost_equal(verts_c1.min(), -custom_outer_radius_1, decimal=7)
    npt.assert_almost_equal(verts_c1.max(), custom_outer_radius_1, decimal=7)
    npt.assert_array_equal(verts_c1[:, 2], np.zeros(verts_c1.shape[0]))
    npt.assert_equal(
        list(set(np.concatenate(faces_c1, axis=None))), list(range(len(verts_c1)))
    )

    # Test with custom parameters: different radii
    custom_inner_radius_2 = 10.0
    custom_outer_radius_2 = 20.0
    custom_radial_segments_2 = 1
    custom_circumferential_segments_2 = 8

    verts_c2, faces_c2 = fp.prim_ring(
        inner_radius=custom_inner_radius_2,
        outer_radius=custom_outer_radius_2,
        radial_segments=custom_radial_segments_2,
        circumferential_segments=custom_circumferential_segments_2,
    )

    expected_verts_shape_c2 = (
        custom_circumferential_segments_2 * (custom_radial_segments_2 + 1),
        3,
    )
    expected_faces_shape_c2 = (
        custom_circumferential_segments_2 * custom_radial_segments_2 * 2,
        3,
    )

    npt.assert_equal(verts_c2.shape, expected_verts_shape_c2)
    npt.assert_equal(faces_c2.shape, expected_faces_shape_c2)
    npt.assert_almost_equal(verts_c2.min(), -custom_outer_radius_2, decimal=7)
    npt.assert_almost_equal(verts_c2.max(), custom_outer_radius_2, decimal=7)
    npt.assert_array_equal(verts_c2[:, 2], np.zeros(verts_c2.shape[0]))
    npt.assert_equal(
        list(set(np.concatenate(faces_c2, axis=None))), list(range(len(verts_c2)))
    )


def test_normalize_geom_param():
    import pytest

    # Scalar float broadcasts to all centers
    result = fp._normalize_geom_param(0.5, 3, "test")
    npt.assert_array_equal(result, [0.5, 0.5, 0.5])
    assert result.shape == (3,)

    # Scalar int
    result = fp._normalize_geom_param(2, 4, "test")
    npt.assert_array_equal(result, [2.0, 2.0, 2.0, 2.0])

    # Numpy scalar
    result = fp._normalize_geom_param(np.float64(1.5), 2, "test")
    npt.assert_array_equal(result, [1.5, 1.5])

    # Single-element list broadcasts
    result = fp._normalize_geom_param([0.7], 3, "test")
    npt.assert_array_equal(result, [0.7, 0.7, 0.7])

    # Single-element tuple broadcasts
    result = fp._normalize_geom_param((0.7,), 3, "test")
    npt.assert_array_equal(result, [0.7, 0.7, 0.7])

    # Single-element ndarray broadcasts
    result = fp._normalize_geom_param(np.array([0.7]), 3, "test")
    npt.assert_array_equal(result, [0.7, 0.7, 0.7])

    # Array of correct length returned as-is
    result = fp._normalize_geom_param(np.array([1.0, 2.0, 3.0]), 3, "test")
    npt.assert_array_equal(result, [1.0, 2.0, 3.0])

    # List of correct length
    result = fp._normalize_geom_param([1.0, 2.0], 2, "test")
    npt.assert_array_equal(result, [1.0, 2.0])

    # N=1 with scalar
    result = fp._normalize_geom_param(0.5, 1, "test")
    npt.assert_array_equal(result, [0.5])

    # Wrong-size array raises ValueError with param name
    with pytest.raises(ValueError, match="my_param"):
        fp._normalize_geom_param(np.array([1.0, 2.0]), 3, "my_param")

    with pytest.raises(ValueError, match="radius"):
        fp._normalize_geom_param([1.0, 2.0, 3.0], 5, "radius")

    # dtype is float64
    result = fp._normalize_geom_param(1, 2, "test")
    assert result.dtype == np.float64


def test_prim_ring_error_handling():
    with npt.assert_raises(ValueError):
        fp.prim_ring(radial_segments=0)

    with npt.assert_raises(ValueError):
        fp.prim_ring(circumferential_segments=2)

    with npt.assert_raises(ValueError):
        fp.prim_ring(inner_radius=1.0, outer_radius=0.5)

    with npt.assert_raises(ValueError):
        fp.prim_ring(inner_radius=1.0, outer_radius=1.0)

    with npt.assert_raises(ValueError):
        fp.prim_ring(inner_radius=-0.5, outer_radius=0.0)

    with npt.assert_raises(ValueError):
        fp.prim_ring(inner_radius=0.0, outer_radius=-1.0)
