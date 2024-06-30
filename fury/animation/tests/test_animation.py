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
from fury.lib import Camera


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_animation():
    shaders = False
    anim = Animation()

    cube_actor = actor.cube(np.array([[0, 0, 0]]))
    anim.add(cube_actor)

    assert cube_actor in anim.actors
    assert cube_actor not in anim.static_actors

    anim.add_static_actor(cube_actor)
    assert cube_actor in anim.static_actors

    anim = Animation(actors=cube_actor)
    assert cube_actor in anim.actors

    anim_main = Animation()
    anim_main.add_child_animation(anim)
    assert anim in anim_main.child_animations

    anim = Animation(actors=cube_actor)
    anim.set_position(0, np.array([1, 1, 1]))
    # overriding a keyframe
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(3, np.array([2, 2, 2]))
    anim.set_position(5, np.array([3, 15, 2]))
    anim.set_position(7, np.array([4, 2, 20]))

    anim.set_opacity(0, 0)
    anim.set_opacity(7, 1)

    anim.set_rotation(0, np.array([90, 0, 0]))
    anim.set_rotation(7, np.array([0, 180, 0]))

    anim.set_scale(0, np.array([1, 1, 1]))
    anim.set_scale(7, np.array([5, 5, 5]))

    anim.set_color(0, np.array([1, 0, 1]))

    npt.assert_almost_equal(anim.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(anim.get_position(7), np.array([4, 2, 20]))

    anim.set_position_interpolator(linear_interpolator)
    anim.set_position_interpolator(cubic_bezier_interpolator)
    anim.set_position_interpolator(step_interpolator)
    anim.set_position_interpolator(cubic_spline_interpolator)
    anim.set_position_interpolator(spline_interpolator, degree=2)
    anim.set_rotation_interpolator(step_interpolator)
    anim.set_scale_interpolator(linear_interpolator)
    anim.set_opacity_interpolator(step_interpolator)
    anim.set_color_interpolator(linear_interpolator)

    npt.assert_almost_equal(anim.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(anim.get_position(7), np.array([4, 2, 20]))

    npt.assert_almost_equal(anim.get_color(7), np.array([1, 0, 1]))
    anim.set_color(25, np.array([0.2, 0.2, 0.5]))
    assert_not_equal(anim.get_color(7), np.array([1, 0, 1]))
    assert_not_equal(anim.get_color(25), np.array([0.2, 0.2, 0.5]))

    cube = actor.cube(np.array([[0, 0, 0]]))
    anim.add_actor(cube)
    anim.update_animation(time=0)
    if not shaders:
        transform = cube.GetUserTransform()
        npt.assert_almost_equal(anim.get_position(0), transform.GetPosition())
        npt.assert_almost_equal(anim.get_scale(0), transform.GetScale())
        npt.assert_almost_equal(anim.get_rotation(0), transform.GetOrientation())


def test_camera_animation():
    cam = Camera()
    anim = CameraAnimation(camera=cam)

    assert anim.camera is cam

    anim.set_position(0, [1, 2, 3])
    anim.set_position(3, [3, 2, 1])

    anim.set_focal(0, [10, 20, 30])
    anim.set_focal(3, [30, 20, 10])

    anim.set_rotation(0, np.array([180, 0, 0]))

    anim.update_animation(time=0)
    npt.assert_almost_equal(cam.GetPosition(), np.array([1, 2, 3]))
    npt.assert_almost_equal(cam.GetFocalPoint(), np.array([10, 20, 30]))
    anim.update_animation(time=3)
    npt.assert_almost_equal(cam.GetPosition(), np.array([3, 2, 1]))
    npt.assert_almost_equal(cam.GetFocalPoint(), np.array([30, 20, 10]))
    anim.update_animation(time=1.5)
    npt.assert_almost_equal(cam.GetPosition(), np.array([2, 2, 2]))
    npt.assert_almost_equal(cam.GetFocalPoint(), np.array([20, 20, 20]))
    rot = np.zeros(16)
    matrix = cam.GetModelTransformMatrix()
    matrix.DeepCopy(rot.ravel(), matrix)
    expected = np.array([[1, 0, 0, 0], [0, -1, 0, 4], [0, 0, -1, 2], [0, 0, 0, 1]])
    npt.assert_almost_equal(expected, rot.reshape([4, 4]))
