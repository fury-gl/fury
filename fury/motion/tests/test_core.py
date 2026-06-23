import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import transform

from fury.actor import box
from fury.lib import PerspectiveCamera
from fury.motion import Animation, CameraAnimation
from fury.motion.helpers import compose_transform_matrix
from fury.motion.interpolator import (
    cubic_bezier_interpolator,
    cubic_spline_interpolator,
    linear_interpolator,
    spline_interpolator,
    step_interpolator,
)


@pytest.fixture
def sample_actor():
    return box(
        np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        scales=np.array([[1, 1, 1]]),
    )


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_animation_record_requires_show_manager():
    anim = Animation()

    with pytest.raises(RuntimeError, match="ShowManager"):
        anim.record("animation.mp4")


def test_animation_creation(sample_actor):
    anim = Animation()

    anim.add(sample_actor)
    assert sample_actor in anim.actors
    assert sample_actor not in anim.static_actors

    anim.add_static_actor(sample_actor)
    assert sample_actor in anim.static_actors

    anim2 = Animation(actors=sample_actor)
    assert sample_actor in anim2.actors

    anim_main = Animation()
    anim_main.add_child_animation(anim)
    assert anim in anim_main.child_animations


def test_animation_keyframes():
    anim = Animation()

    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(3, np.array([2, 2, 2]))
    anim.set_position(5, np.array([3, 15, 2]))
    anim.set_position(7, np.array([4, 2, 20]))

    npt.assert_almost_equal(anim.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(anim.get_position(7), np.array([4, 2, 20]))

    anim.set_position(0, np.array([1, 1, 1]))
    npt.assert_almost_equal(anim.get_position(0), np.array([1, 1, 1]))

    anim.set_opacity(0, 0.0)
    anim.set_opacity(7, 1.0)
    npt.assert_almost_equal(anim.get_opacity(0), 0.0)
    npt.assert_almost_equal(anim.get_opacity(7), 1.0)

    anim.set_rotation(0, np.array([90, 0, 0]))
    anim.set_rotation(7, np.array([0, 180, 0]))

    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(7, np.array([5, 5, 5]))
    npt.assert_almost_equal(anim.get_scale(0), np.array([1, 1, 1]))
    npt.assert_almost_equal(anim.get_scale(7), np.array([5, 5, 5]))

    anim.set_color(0, np.array([1, 0, 1]))
    anim.set_color(7, np.array([0.2, 0.2, 0.5]))


def test_animation_interpolators():
    anim = Animation()

    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(3, np.array([2, 1, 10]))
    anim.set_position(5, np.array([3, 2, 15]))
    anim.set_position(7, np.array([4, 2, 20]))

    anim.set_position_interpolator(linear_interpolator)
    npt.assert_almost_equal(anim.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(anim.get_position(7), np.array([4, 2, 20]))

    anim.set_position_interpolator(cubic_bezier_interpolator)
    anim.set_position_interpolator(step_interpolator)
    anim.set_position_interpolator(cubic_spline_interpolator)
    anim.set_position_interpolator(spline_interpolator, degree=2)

    anim.set_rotation(0, np.array([0, 0, 0]))
    anim.set_rotation(7, np.array([90, 0, 0]))
    anim.set_rotation_interpolator(step_interpolator)

    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(7, np.array([2, 2, 2]))
    anim.set_scale_interpolator(linear_interpolator)

    anim.set_opacity(0, 0.0)
    anim.set_opacity(7, 1.0)
    anim.set_opacity_interpolator(step_interpolator)

    anim.set_color(0, np.array([1, 0, 0]))
    anim.set_color(7, np.array([0, 1, 0]))
    anim.set_color_interpolator(linear_interpolator)


def test_compose_transform_matrix():
    matrix = compose_transform_matrix(
        position=np.array([1, 2, 3]),
        rotation_quat=None,
        scale_factors=np.array([2, 3, 4]),
    )

    npt.assert_almost_equal(matrix[:3, 3], np.array([1, 2, 3]))
    npt.assert_almost_equal(np.diag(matrix)[:3], np.array([2, 3, 4]))


def test_compose_transform_matrix_with_parent_rotation():
    parent = compose_transform_matrix(
        rotation_quat=transform.Rotation.from_euler("z", 90, degrees=True).as_quat()
    )
    matrix = compose_transform_matrix(
        position=np.array([1, 0, 0]), parent_matrix=parent
    )

    npt.assert_almost_equal(matrix[:3, 3], np.array([0, 1, 0]), decimal=6)


def test_animation_update_applies_pygfx_actor_state(sample_actor):
    anim = Animation(actors=sample_actor)

    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(2, np.array([10, 10, 10]))
    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(2, np.array([2, 2, 2]))
    anim.set_opacity(0, 0.0)
    anim.set_opacity(2, 1.0)
    anim.set_color(0, np.array([1, 0, 0]))
    anim.set_color(2, np.array([0, 1, 0]))

    anim.update_animation(time=1)

    npt.assert_almost_equal(sample_actor.local.matrix[:3, 3], np.array([5, 5, 5]))
    npt.assert_almost_equal(
        np.diag(sample_actor.local.matrix)[:3], np.array([1.5, 1.5, 1.5])
    )
    npt.assert_almost_equal(sample_actor.material.opacity, 0.5)
    colors = sample_actor.geometry.colors.data[:, :3]
    npt.assert_almost_equal(colors, np.repeat([[0.5, 0.5, 0.0]], len(colors), axis=0))


def test_animation_duration():
    anim = Animation()

    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([1, 1, 1]))
    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(10, np.array([2, 2, 2]))

    anim.update_duration()
    assert anim.duration == 10

    anim2 = Animation(length=20)
    anim2.set_position(0, np.array([0, 0, 0]))
    anim2.set_position(5, np.array([1, 1, 1]))
    anim2.update_duration()
    assert anim2.duration == 20


def test_animation_loop():
    anim = Animation(loop=True)
    assert anim.loop is True

    anim.loop = False
    assert anim.loop is False


def test_animation_child_animations_inherit_parent_transform(sample_actor):
    from fury.actor import box

    parent_actor = sample_actor
    child_actor = box(
        np.array([[0, 0, 0]]),
        colors=np.array([[0, 1, 0]]),
        scales=np.array([[1, 1, 1]]),
    )
    parent = Animation(actors=parent_actor)
    child = Animation(actors=child_actor)

    parent.set_position(0, np.array([1, 2, 3]))
    parent.set_rotation(0, np.array([0, 0, 90]))
    child.set_position(0, np.array([4, 0, 0]))
    parent.add_child_animation(child)

    parent.update_animation(time=0)

    npt.assert_almost_equal(child_actor.local.matrix[:3, 3], np.array([1, 6, 3]))


def test_animation_callbacks():
    anim = Animation()
    callback_values = []

    def test_callback(t):
        callback_values.append(t)

    anim.add_update_callback(test_callback)

    anim.update_animation(time=0)
    anim.update_animation(time=1)
    anim.update_animation(time=2)

    assert callback_values == [0, 1, 2]


def test_camera_animation_position():
    camera = PerspectiveCamera()
    anim = CameraAnimation(camera=camera)

    assert anim.camera is camera

    anim.set_position(0, np.array([1, 2, 3]))
    anim.set_position(2, np.array([3, 2, 1]))
    anim.update_animation(time=1)

    npt.assert_almost_equal(camera.local.position, np.array([2, 2, 2]))


def test_camera_animation_focal_and_view_up():
    camera = PerspectiveCamera()
    anim = CameraAnimation(camera=camera)

    anim.set_position(0, np.array([0, 0, 10]))
    anim.set_focal(0, np.array([0, 0, 0]))
    anim.set_focal(2, np.array([2, 0, 0]))
    anim.set_view_up(0, np.array([0, 1, 0]))
    anim.set_view_up(2, np.array([0, 0, 1]))

    anim.update_animation(time=1)

    npt.assert_almost_equal(camera.local.position, np.array([0, 0, 10]))
    expected_up = np.array([0, 0.5, 0.5])
    expected_up = expected_up / np.linalg.norm(expected_up)
    npt.assert_almost_equal(camera.local.reference_up, expected_up)


def test_camera_animation_rotation():
    camera = PerspectiveCamera()
    anim = CameraAnimation(camera=camera)

    anim.set_position(0, np.array([1, 2, 3]))
    anim.set_rotation(0, np.array([0, 0, 90]))
    anim.update_animation(time=0)

    expected = transform.Rotation.from_euler(
        "zxy", np.array([90, 0, 0]), degrees=True
    ).as_matrix()
    npt.assert_almost_equal(camera.local.matrix[:3, :3], expected, decimal=6)
    npt.assert_almost_equal(camera.local.matrix[:3, 3], np.array([1, 2, 3]))


def test_camera_animation_without_camera_is_noop():
    anim = CameraAnimation()

    anim.set_position(0, np.array([1, 2, 3]))
    anim.update_animation(time=0)

    npt.assert_almost_equal(anim.current_timestamp, 0)


def test_camera_animation_update_before_add_to_scene():
    camera = PerspectiveCamera()
    anim = CameraAnimation(camera=camera)

    anim.set_position(0, np.array([1, 2, 3]))
    anim.update_animation()

    assert anim.current_timestamp < 1
