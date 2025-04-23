from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import actor, window


def validate_actors(actor_type="actor_name", prim_count=1, **kwargs):
    scene = window.Scene()
    typ_actor = getattr(actor, actor_type)
    get_actor = typ_actor(**kwargs)
    scene.add(get_actor)

    centers = kwargs.get("centers", None)
    colors = kwargs.get("colors", None)

    if centers is not None:
        npt.assert_array_equal(get_actor.local.position, centers[0])

        mean_vertex = np.round(np.mean(get_actor.geometry.positions.view, axis=0))
        npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert get_actor.prim_count == prim_count

    if actor_type == "line":
        return

    fname = f"{actor_type}_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(get_actor)

    typ_actor_1 = getattr(actor, actor_type)
    get_actor_1 = typ_actor_1(centers=centers, colors=colors, material="basic")
    scene.add(get_actor_1)
    fname_1 = f"{actor_type}_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == 0 and b == 0
    assert r == 255
    scene.remove(get_actor_1)


def test_sphere():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="sphere")


def test_line():
    lines_points = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]])
    colors = np.array([[[1, 0, 0]], [[0, 1, 0]]])
    validate_actors(lines=lines_points, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    colors = None
    validate_actors(lines=line, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    actor.line(line, colors=colors)
    actor.line(line)
    actor.line(line, colors=colors)
    actor.line(line, colors=colors, material="basic")
    actor.line(line, colors=line, material="basic")


def test_box():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="box")


def test_cylinder():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cylinder")


def test_square():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="square")


def test_frustum():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="frustum")


def test_tetrahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="tetrahedron")


def test_icosahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="icosahedron")


def test_rhombicuboctahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="rhombicuboctahedron")


def test_triangularprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="triangularprism")


def test_pentagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="pentagonalprism")


def test_octagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="octagonalprism")


def test_arrow():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="arrow")


def test_superquadric():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="superquadric")


def test_cone():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cone")


def test_star():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="star")


def test_flat_disk():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="disk")
    validate_actors(centers=centers, colors=colors, actor_type="disk", sectors=8)


def test_point():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()
    point_actor = actor.point(centers=centers, colors=colors)
    scene.add(point_actor)

    npt.assert_array_equal(point_actor.local.position, centers[0])

    mean_vertex = np.round(np.mean(point_actor.geometry.positions.view, axis=0))
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    fname = "point_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    scene.remove(point_actor)

    point_actor_1 = actor.point(centers=centers, colors=colors, material="gaussian")
    scene.add(point_actor_1)
    fname_1 = "point_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    scene.remove(point_actor_1)


def test_marker():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()
    marker_actor = actor.marker(centers=centers, colors=colors)
    scene.add(marker_actor)

    npt.assert_array_equal(marker_actor.local.position, centers[0])

    mean_vertex = np.round(np.mean(marker_actor.geometry.positions.view, axis=0))
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    fname = "marker_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    scene.remove(marker_actor)

    marker_actor_1 = actor.marker(centers=centers, colors=colors, marker="heart")
    scene.add(marker_actor_1)
    fname_1 = "marker_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    scene.remove(marker_actor_1)


def test_text():
    text = "FURY"
    position = np.array([0, 0, 0])
    scene = window.Scene()

    text_actor = actor.text(text=text, anchor="middle-center")
    scene.add(text_actor)

    npt.assert_array_equal(text_actor.local.position, position)

    fname = "text_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r == mean_b and mean_r == mean_g
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(text_actor)

    text1 = "HELLO"
    text_actor_1 = actor.text(text=text1, anchor="middle-center")
    text_actor_1.local.position = position
    scene.add(text_actor_1)
    fname_1 = "text_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r == mean_b and mean_r == mean_g
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(text_actor_1)


def test_axes():
    scene = window.Scene()
    axes_actor = actor.axes()
    scene.add(axes_actor)

    assert axes_actor.prim_count == 3

    fname = "axes_test.png"
    window.snapshot(scene=scene, fname=fname)
    img = Image.open(fname)
    img_array = np.array(img)
    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )
    assert np.isclose(mean_r, mean_g, atol=0.02)
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(axes_actor)


def test_ellipsoid():
    centers = np.array([[0, 0, 0]])
    lengths = np.array([[2, 1, 1]])
    axes = np.array([np.eye(3)])
    colors = np.array([1, 0, 0])

    validate_actors(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
        actor_type="ellipsoid",
    )

    _ = actor.ellipsoid(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]),
        lengths=np.array([[2, 1, 1]]),
        colors=np.array([[1, 0, 0]]),
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]), lengths=(2, 1, 1), colors=(1, 0, 0)
    )

    _ = actor.ellipsoid(centers)
