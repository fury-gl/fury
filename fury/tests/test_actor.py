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

    # window.record(scene=scene, fname="sphere_test_1.png")

    # img = Image.open("sphere_test_1.png")
    # img_array = np.array(img)

    # mean_r, mean_g, mean_b, _ = np.mean(
    #     img_array.reshape(-1, img_array.shape[2]), axis=0
    # )

    # assert mean_r > mean_b and mean_r > mean_g
    # assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255

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

    # TODO: Record does not exist in the v2, test needs to be update
    # window.record(scene=scene, fname="sphere_test_2.png")

    # img = Image.open("sphere_test_2.png")
    # img_array = np.array(img)

    # mean_r, mean_g, mean_b, mean_a = np.mean(
    #     img_array.reshape(-1, img_array.shape[2]), axis=0
    # )

    # assert mean_r > mean_b and mean_r > mean_g
    # assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255
    # assert mean_a == 255.0
    # assert mean_b == 0.0
    # assert mean_g == 0.0

    vertices = sphere_actor_2.geometry.positions.view
    faces = sphere_actor_2.geometry.indices.view
    colors = sphere_actor_2.geometry.colors.view

    vertices_mean = np.mean(vertices, axis=0)

    npt.assert_array_almost_equal(vertices_mean, centers[0])

    assert len(vertices) == len(colors)

    npt.assert_array_almost_equal(len(faces), (2 * phi * (theta - 2)))
