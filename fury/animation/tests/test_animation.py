"""Tests for Animation and CameraAnimation classes (pygfx backend)."""

import numpy as np
import numpy.testing as npt

from fury import actor
from fury.animation import Animation, CameraAnimation
from fury.animation.interpolator import (
    cubic_bezier_interpolator,
    cubic_spline_interpolator,
    linear_interpolator,
    spline_interpolator,
    step_interpolator,
)


def assert_not_equal(x, y):
    """Assert that two arrays are not equal."""
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_animation_creation():
    """Test Animation object creation and actor management."""
    anim = Animation()

    box_actor = actor.box(np.array([[0, 0, 0]]))
    anim.add_actor(box_actor)

    assert box_actor in anim.actors
    assert box_actor not in anim.static_actors

    anim.add_static_actor(box_actor)
    assert box_actor in anim.static_actors

    anim2 = Animation(actors=box_actor)
    assert box_actor in anim2.actors

    anim_main = Animation()
    anim_main.add_child_animation(anim)
    assert anim in anim_main.child_animations


def test_animation_keyframes():
    """Test Animation keyframe setting and getting."""
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
    """Test setting different interpolators on animation properties."""
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


def test_animation_update():
    """Test Animation update with pygfx actor transforms."""
    box_actor = actor.box(np.array([[0, 0, 0]]))
    anim = Animation(actors=box_actor)

    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(2, np.array([10, 10, 10]))

    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(2, np.array([2, 2, 2]))

    anim.update_animation(time=0)
    npt.assert_almost_equal(box_actor.local.position, np.array([0, 0, 0]))

    anim.update_animation(time=2)
    npt.assert_almost_equal(box_actor.local.position, np.array([10, 10, 10]))

    anim.update_animation(time=1)
    npt.assert_almost_equal(box_actor.local.position, np.array([5, 5, 5]))


def test_animation_duration():
    """Test Animation duration calculation."""
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
    """Test Animation loop behavior."""
    anim = Animation(loop=True)
    assert anim.loop is True

    anim.loop = False
    assert anim.loop is False


def test_camera_animation():
    """Test CameraAnimation with pygfx camera."""
    from pygfx import PerspectiveCamera

    cam = PerspectiveCamera()
    anim = CameraAnimation(camera=cam)

    assert anim.camera is cam

    anim.set_position(0, np.array([1, 2, 3]))
    anim.set_position(3, np.array([3, 2, 1]))

    anim.set_focal(0, np.array([10, 20, 30]))
    anim.set_focal(3, np.array([30, 20, 10]))

    anim.update_animation(time=0)
    npt.assert_almost_equal(cam.local.position, np.array([1, 2, 3]))

    anim.update_animation(time=3)
    npt.assert_almost_equal(cam.local.position, np.array([3, 2, 1]))

    anim.update_animation(time=1.5)
    npt.assert_almost_equal(cam.local.position, np.array([2, 2, 2]))


def test_camera_animation_focal_keyframes():
    """Test CameraAnimation focal position keyframes."""
    anim = CameraAnimation()

    focal_positions = {
        0: np.array([0, 0, 0]),
        5: np.array([10, 10, 10]),
        10: np.array([20, 20, 20]),
    }

    anim.set_focal_keyframes(focal_positions)

    npt.assert_almost_equal(anim.get_focal(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(anim.get_focal(10), np.array([20, 20, 20]))


def test_camera_animation_view_up():
    """Test CameraAnimation view up direction."""
    anim = CameraAnimation()

    anim.set_view_up(0, np.array([0, 1, 0]))
    anim.set_view_up(5, np.array([0, 0, 1]))

    npt.assert_almost_equal(anim.get_view_up(0), np.array([0, 1, 0]))
    npt.assert_almost_equal(anim.get_view_up(5), np.array([0, 0, 1]))


def test_animation_child_animations():
    """Test Animation with child animations."""
    parent = Animation()
    child1 = Animation()
    child2 = Animation()

    parent.add_child_animation(child1)
    parent.add_child_animation(child2)

    assert child1 in parent.child_animations
    assert child2 in parent.child_animations
    assert len(parent.child_animations) == 2


def test_animation_callbacks():
    """Test Animation general callbacks."""
    anim = Animation()
    callback_values = []

    def test_callback(t):
        callback_values.append(t)

    anim.add_update_callback(test_callback)

    anim.update_animation(time=0)
    anim.update_animation(time=1)
    anim.update_animation(time=2)

    assert len(callback_values) == 3
    assert callback_values == [0, 1, 2]
