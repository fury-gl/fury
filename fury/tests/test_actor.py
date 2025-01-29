from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import actor, window


def test_sphere():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    radii = np.array([1])

    sphere_actor = actor.sphere(centers=centers, colors=colors, radii=radii)
    scene.add(sphere_actor)

    window.snapshot(scene=scene, fname="sphere_test_1.png")

    img = Image.open("sphere_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _ = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255

    npt.assert_array_equal(sphere_actor.local.position, centers[0])

    assert sphere_actor.prim_count == 1

    # center_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    # npt.assert_array_equal(center_pixel[0], colors[0][0] * 255)

    phi, theta = 100, 100
    sphere_actor_2 = actor.sphere(
        centers=centers,
        colors=colors,
        radii=radii,
        opacity=1,
        material="basic",
        phi=phi,
        theta=theta,
    )
    scene.remove(sphere_actor)
    scene.add(sphere_actor_2)

    window.snapshot(scene=scene, fname="sphere_test_2.png")

    img = Image.open("sphere_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255
    assert mean_a == 255.0
    assert mean_b == 0.0
    assert mean_g == 0.0

    vertices = sphere_actor_2.geometry.positions.view
    faces = sphere_actor_2.geometry.indices.view
    colors = sphere_actor_2.geometry.colors.view

    vertices_mean = np.mean(vertices, axis=0)

    npt.assert_array_almost_equal(vertices_mean, centers[0])

    assert len(vertices) == len(colors)

    npt.assert_array_almost_equal(len(faces), (2 * phi * (theta - 2)))


def test_box():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scales = np.array([[1, 1, 7]])

    box_actor = actor.box(centers=centers, colors=colors, scales=scales)
    scene.add(box_actor)

    npt.assert_array_equal(box_actor.local.position, centers[0])

    mean_vertex = np.mean(box_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert box_actor.prim_count == 1

    scene.remove(box_actor)


def test_cylinder():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    sectors = 36
    capped = True

    cylinder_actor = actor.cylinder(
        centers=centers, colors=colors, sectors=sectors, capped=capped
    )
    scene.add(cylinder_actor)

    npt.assert_array_equal(cylinder_actor.local.position, centers[0])

    mean_vertex = np.mean(cylinder_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0], decimal=2)

    assert cylinder_actor.prim_count == 1

    window.snapshot(scene=scene, fname="cylinder_test_1.png")

    img = Image.open("cylinder_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255 and 0 < mean_g < 255 and 0 <= mean_b < 255

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    assert r > 0 and g > 0 and b > 0

    scene.remove(cylinder_actor)

    cylinder_actor_2 = actor.cylinder(
        centers=centers, colors=colors, sectors=sectors, capped=capped, material="basic"
    )
    scene.add(cylinder_actor_2)
    window.snapshot(scene=scene, fname="cylinder_test_2.png")

    img = Image.open("cylinder_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(cylinder_actor_2)


def test_square():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    square_actor = actor.square(centers=centers, colors=colors)
    scene.add(square_actor)

    npt.assert_array_equal(square_actor.local.position, centers[0])

    mean_vertex = np.mean(square_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert square_actor.prim_count == 1
    scene.remove(square_actor)


def test_frustum():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    frustum_actor = actor.frustum(centers=centers, colors=colors)
    scene.add(frustum_actor)

    npt.assert_array_equal(frustum_actor.local.position, centers[0])

    mean_vertex = np.mean(frustum_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert frustum_actor.prim_count == 1

    window.snapshot(scene=scene, fname="frustum_test_1.png")

    img = Image.open("frustum_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255 and 0 < mean_g < 255 and 0 <= mean_b < 255

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    assert r > 0 and g > 0 and b > 0
    scene.remove(frustum_actor)

    frustum_actor_2 = actor.frustum(centers=centers, colors=colors, material="basic")
    scene.add(frustum_actor_2)
    window.snapshot(scene=scene, fname="frustum_test_2.png")

    img = Image.open("frustum_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(frustum_actor_2)


def test_tetrahedron():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    tetrahedron_actor = actor.tetrahedron(centers=centers, colors=colors)
    scene.add(tetrahedron_actor)

    npt.assert_array_equal(tetrahedron_actor.local.position, centers[0])

    mean_vertex = np.mean(tetrahedron_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert tetrahedron_actor.prim_count == 1

    window.snapshot(scene=scene, fname="tetrahedron_test_1.png")

    img = Image.open("tetrahedron_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(tetrahedron_actor)

    tetrahedron_actor_2 = actor.tetrahedron(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(tetrahedron_actor_2)
    window.snapshot(scene=scene, fname="tetrahedron_test_2.png")

    img = Image.open("tetrahedron_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(tetrahedron_actor_2)


def test_icosahedron():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    icosahedron_actor = actor.icosahedron(centers=centers, colors=colors)
    scene.add(icosahedron_actor)

    npt.assert_array_equal(icosahedron_actor.local.position, centers[0])

    mean_vertex = np.mean(icosahedron_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert icosahedron_actor.prim_count == 1

    window.snapshot(scene=scene, fname="icosahedron_test_1.png")

    img = Image.open("icosahedron_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(icosahedron_actor)

    icosahedron_actor_2 = actor.icosahedron(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(icosahedron_actor_2)
    window.snapshot(scene=scene, fname="icosahedron_test_2.png")

    img = Image.open("icosahedron_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(icosahedron_actor_2)


def test_rhombicuboctahedron():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    rhombicuboctahedron_actor = actor.rhombicuboctahedron(
        centers=centers, colors=colors
    )
    scene.add(rhombicuboctahedron_actor)

    npt.assert_array_equal(rhombicuboctahedron_actor.local.position, centers[0])

    mean_vertex = np.mean(rhombicuboctahedron_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert rhombicuboctahedron_actor.prim_count == 1

    window.snapshot(scene=scene, fname="rhombicuboctahedron_test_1.png")

    img = Image.open("rhombicuboctahedron_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(rhombicuboctahedron_actor)

    rhombicuboctahedron_actor_2 = actor.rhombicuboctahedron(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(rhombicuboctahedron_actor_2)
    window.snapshot(scene=scene, fname="rhombicuboctahedron_test_2.png")

    img = Image.open("rhombicuboctahedron_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(rhombicuboctahedron_actor_2)


def test_triangularprism():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    triangularprism_actor = actor.triangularprism(centers=centers, colors=colors)
    scene.add(triangularprism_actor)

    npt.assert_array_equal(triangularprism_actor.local.position, centers[0])

    mean_vertex = np.mean(triangularprism_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert triangularprism_actor.prim_count == 1

    window.snapshot(scene=scene, fname="triangularprism_test_1.png")

    img = Image.open("triangularprism_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(triangularprism_actor)

    triangularprism_actor_2 = actor.triangularprism(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(triangularprism_actor_2)
    window.snapshot(scene=scene, fname="triangularprism_test_2.png")

    img = Image.open("triangularprism_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(triangularprism_actor_2)


def test_pentagonalprism():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    pentagonalprism_actor = actor.pentagonalprism(centers=centers, colors=colors)
    scene.add(pentagonalprism_actor)

    npt.assert_array_equal(pentagonalprism_actor.local.position, centers[0])

    mean_vertex = np.mean(pentagonalprism_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert pentagonalprism_actor.prim_count == 1

    window.snapshot(scene=scene, fname="pentagonalprism_test_1.png")

    img = Image.open("pentagonalprism_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(pentagonalprism_actor)

    pentagonalprism_actor_2 = actor.pentagonalprism(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(pentagonalprism_actor_2)
    window.snapshot(scene=scene, fname="pentagonalprism_test_2.png")

    img = Image.open("pentagonalprism_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(pentagonalprism_actor_2)


def test_octagonalprism():
    scene = window.Scene()
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])

    octagonalprism_actor = actor.octagonalprism(centers=centers, colors=colors)
    scene.add(octagonalprism_actor)

    npt.assert_array_equal(octagonalprism_actor.local.position, centers[0])

    mean_vertex = np.mean(octagonalprism_actor.geometry.positions.view, axis=0)
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert octagonalprism_actor.prim_count == 1

    window.snapshot(scene=scene, fname="octagonalprism_test_1.png")

    img = Image.open("octagonalprism_test_1.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b
    scene.remove(octagonalprism_actor)

    octagonalprism_actor_2 = actor.octagonalprism(
        centers=centers, colors=colors, material="basic"
    )
    scene.add(octagonalprism_actor_2)
    window.snapshot(scene=scene, fname="octagonalprism_test_2.png")

    img = Image.open("octagonalprism_test_2.png")
    img_array = np.array(img)

    mean_r, mean_g, mean_b, mean_a = np.mean(
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
    scene.remove(octagonalprism_actor_2)
