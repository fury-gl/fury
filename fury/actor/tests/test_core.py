from PIL import Image
import numpy as np
import pytest

from fury import actor, geometry, material, window
from fury.actor import Actor, Image as ActorImage, Text
from fury.actor.tests._helpers import validate_actors


def test_actor_rotate_sets_quaternion_on_local_rotation(sphere_actor):
    sphere_actor.rotate((90, 0, 0))

    assert sphere_actor.local.rotation is not None
    np.testing.assert_array_almost_equal(
        sphere_actor.local.rotation,
        Actor._euler_to_quaternion(np.radians([90, 0, 0])),
    )


@pytest.mark.parametrize(
    "invalid_rotation",
    [
        (0, 0),  # too short
        (0, 0, 0, 0),  # too long
    ],
)
def test_actor_rotate_invalid_input_raises(sphere_actor, invalid_rotation):
    with pytest.raises(ValueError, match="Rotation must contain three angles"):
        sphere_actor.rotate(invalid_rotation)


def test_actor_translate_updates_local_translation_vector(sphere_actor):
    sphere_actor.translate((1, 2, 3))

    np.testing.assert_array_equal(
        sphere_actor.local.position, np.array([1, 2, 3], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "scale_input,expected",
    [
        (2, np.array([2, 2, 2], dtype=np.float32)),
        ((1, 2, 3), np.array([1, 2, 3], dtype=np.float32)),
    ],
)
def test_actor_scale_accepts_scalar_and_vector_inputs(
    sphere_actor, scale_input, expected
):
    sphere_actor.scale(scale_input)

    np.testing.assert_array_equal(sphere_actor.local.scale, expected)


@pytest.mark.parametrize(
    "invalid_scale",
    [
        (1, 2),
        (1, 2, 3, 4),
    ],
)
def test_actor_scale_invalid_shape(sphere_actor, invalid_scale):
    with pytest.raises(ValueError, match="Scale must contain three values"):
        sphere_actor.scale(invalid_scale)


def test_actor_transform_accepts_4x4_matrix(sphere_actor):
    matrix = np.eye(4, dtype=np.float32)

    sphere_actor.transform(matrix)

    np.testing.assert_array_equal(sphere_actor.local.matrix, matrix)


@pytest.mark.parametrize(
    "invalid_matrix",
    [
        np.eye(3),
        np.ones((4, 3)),
        np.zeros((5, 5)),
    ],
)
def test_actor_transform_invalid_matrix_shape(sphere_actor, invalid_matrix):
    with pytest.raises(ValueError, match="Transformation matrix must be of shape"):
        sphere_actor.transform(invalid_matrix)


@pytest.mark.parametrize(
    "opacity",
    [0, 0.5, 1],
)
def test_actor_opacity_sets_material(sphere_actor, opacity):
    sphere_actor.opacity = opacity
    assert sphere_actor.material.opacity == opacity


@pytest.mark.parametrize(
    "invalid_opacity",
    [
        -0.5,  # negative
        1.5,  # greater than 1
    ],
)
def test_actor_opacity_invalid_values_raise(sphere_actor, invalid_opacity):
    with pytest.raises(ValueError, match="Opacity must be"):
        sphere_actor.opacity = invalid_opacity


def test_create_mesh():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(positions)
    mat = material._create_mesh_material(
        material="phong", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    mesh = actor.create_mesh(geometry=geo, material=mat)
    assert mesh.geometry == geo
    assert mesh.material == mat


def test_create_point():
    vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(vertices)
    mat = material._create_points_material(
        material="basic", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    point = actor.create_point(geometry=geo, material=mat)
    assert point.geometry == geo
    assert point.material == mat


def test_create_text():
    text = "FURY"
    mat = material._create_text_material(color=(1, 0, 0), opacity=0.5)
    text_obj = actor.create_text(text=text, material=mat)
    assert text_obj.material == mat
    assert isinstance(text_obj, Text)


def test_create_image():
    img_data = np.random.rand(128, 128)
    mat = material._create_image_material()
    image_obj = actor.create_image(image_input=img_data, material=mat)
    assert image_obj.material == mat
    assert isinstance(image_obj, ActorImage)
    assert image_obj.geometry.grid.dim == 2


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


def test_arrow():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="arrow")


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
