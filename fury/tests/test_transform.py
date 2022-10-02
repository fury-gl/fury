import numpy as np
import numpy.testing as npt
from scipy.ndimage import center_of_mass

from fury.transform import (sphere2cart, cart2sphere, euler_matrix,
                            _AXES2TUPLE, _TUPLE2AXES, translate,
                            rotate, scale, apply_transformation,
                            transform_from_matrix)
from fury import primitive, window, utils
from fury.testing import assert_greater


def _make_pts():
    """ Make points around sphere quadrants """
    thetas = np.arange(1, 4) * np.pi/4
    phis = np.arange(8) * np.pi/4
    north_pole = (0, 0, 1)
    south_pole = (0, 0, -1)
    points = [north_pole, south_pole]
    for theta in thetas:
        for phi in phis:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append((x, y, z))
    return np.array(points)


def test_sphere_cart():
    sphere_points = _make_pts()
    # test arrays of points
    rs, thetas, phis = cart2sphere(*(sphere_points.T))
    xyz = sphere2cart(rs, thetas, phis)
    npt.assert_array_almost_equal(xyz, sphere_points.T)
    # test radius estimation
    big_sph_pts = sphere_points * 10.4
    rs, thetas, phis = cart2sphere(*big_sph_pts.T)
    npt.assert_array_almost_equal(rs, 10.4)
    xyz = sphere2cart(rs, thetas, phis)
    npt.assert_array_almost_equal(xyz, big_sph_pts.T, decimal=6)
    # test that result shapes match
    x, y, z = big_sph_pts.T
    r, theta, phi = cart2sphere(x[:1], y[:1], z)
    npt.assert_equal(r.shape, theta.shape)
    npt.assert_equal(r.shape, phi.shape)
    x, y, z = sphere2cart(r[:1], theta[:1], phi)
    npt.assert_equal(x.shape, y.shape)
    npt.assert_equal(x.shape, z.shape)
    # test a scalar point
    pt = sphere_points[3]
    r, theta, phi = cart2sphere(*pt)
    xyz = sphere2cart(r, theta, phi)
    npt.assert_array_almost_equal(xyz, pt)

    # Test full circle on x=1, y=1, z=1
    x, y, z = sphere2cart(*cart2sphere(1.0, 1.0, 1.0))
    npt.assert_array_almost_equal((x, y, z), (1.0, 1.0, 1.0))


def test_euler_matrix():
    rotation = euler_matrix(1, 2, 3, 'syxz')
    npt.assert_equal(np.allclose(np.sum(rotation[0]), -1.34786452), True)

    rotation = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    npt.assert_equal(np.allclose(np.sum(rotation[0]), -0.383436184), True)

    ai, aj, ak = (4.0 * np.pi) * (np.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
        _ = euler_matrix(ai, aj, ak, axes)
    for axes in _TUPLE2AXES.keys():
        _ = euler_matrix(ai, aj, ak, axes)


def test_translate():
    x_translate = np.array([1.5, 0.0, 0.0])
    transform = translate(x_translate)
    npt.assert_almost_equal(transform.shape, (4, 4))
    scene = window.Scene()

    verts, triangles = primitive.prim_box()
    cube = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cube)

    verts = apply_transformation(verts, transform)
    cube = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cube)

    y_translate = np.array([0.0, 1.5, 0.0])
    transform = translate(y_translate)
    verts = apply_transformation(verts, transform)
    cube = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cube)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)


def test_scale():
    x_scale = np.array([2.0, 1.0, 1.0])
    transform = scale(x_scale)
    npt.assert_almost_equal(transform.shape, (4, 4))
    scene = window.Scene()

    verts, triangles = primitive.prim_box()
    cube = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cube)
    arr1 = window.snapshot(scene)
    scene.clear()

    verts = apply_transformation(verts, transform)
    cube = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cube)
    arr2 = window.snapshot(scene)
    scene.clear()

    assert_greater(arr2.mean(), arr1.mean())


def test_rotate():
    transform = rotate(np.array([0.707, 0.0, 0.707, 0.0]))  # by 90 degrees
    npt.assert_almost_equal(transform.shape, (4, 4))
    scene = window.Scene()

    verts, triangles = primitive.prim_cone()
    cone = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cone)
    arr1 = window.snapshot(scene)
    scene.clear()

    verts = apply_transformation(verts, transform)
    cone = utils.get_actor_from_primitive(verts, triangles)
    scene.add(cone)
    arr2 = window.snapshot(scene)

    report = window.analyze_snapshot(arr2)
    npt.assert_equal(report.objects, 1)
    assert_greater(center_of_mass(arr2), center_of_mass(arr1))


def test_transform_from_matrix():
    matrix = np.array([[5, 0, 0, 10],
                       [0, 1, 0, -10],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    trans, rot, scale = transform_from_matrix(matrix)
    npt.assert_equal(trans, np.array([10, -10, 0]))
    npt.assert_equal(scale, np.array([5, 1, 1]))

    matrix = np.array([[1,  0,         0,         1],
                       [0,  0.1542515, 0.9880316, 0],
                       [0, -0.9880316, 0.1542515, 0],
                       [0,  0,         0,         1]])
    trans, rot, scale = transform_from_matrix(matrix)
    npt.assert_array_almost_equal(rot,
                                  np.array([81.126612, -1.415926, 0.0, 0.0]))
