import time
import numpy as np
import numpy.testing as npt
from fury import actor
from fury.animation.interpolator import linear_interpolator, \
    step_interpolator, cubic_spline_interpolator, cubic_bezier_interpolator, \
    spline_interpolator
from fury.animation.timeline import Timeline
import fury.testing as ft
from fury.ui import PlaybackPanel


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_timeline():
    shaders = False
    tl = Timeline(playback_panel=True)
    tl.set_position(0, np.array([1, 1, 1]))
    # overriding a keyframe
    tl.set_position(0, np.array([0, 0, 0]))
    tl.set_position(3, np.array([2, 2, 2]))
    tl.set_position(5, np.array([3, 15, 2]))
    tl.set_position(7, np.array([4, 2, 20]))

    tl.set_opacity(0, 0)
    tl.set_opacity(7, 1)

    tl.set_rotation(0, np.array([90, 0, 0]))
    tl.set_rotation(7, np.array([0, 180, 0]))

    tl.set_scale(0, np.array([1, 1, 1]))
    tl.set_scale(7, np.array([5, 5, 5]))

    tl.set_color(0, np.array([1, 0, 1]))

    # test playback panel
    ft.assert_true(isinstance(tl.playback_panel, PlaybackPanel))

    for t in [-10, 0, 2.2, 7, 100]:
        tl.seek(t)
        ft.assert_less_equal(tl.current_timestamp, tl.final_timestamp)
        ft.assert_greater_equal(tl.current_timestamp, 0)

        ft.assert_greater_equal(tl.current_timestamp,
                             tl.playback_panel.current_time)

        if 0 <= t <= tl.final_timestamp:
            npt.assert_almost_equal(tl.current_timestamp, t)
            # check if seeking a certain time affects the time slider's value.
            npt.assert_almost_equal(tl.current_timestamp,
                                    tl.playback_panel.current_time)

    tl.play()
    t_before = tl.current_timestamp
    time.sleep(0.1)
    assert_not_equal(tl.current_timestamp, t_before)
    ft.assert_true(tl.playing)

    tl.pause()
    t_before = tl.current_timestamp
    ft.assert_true(tl.paused)
    time.sleep(0.1)
    npt.assert_almost_equal(tl.current_timestamp, t_before)

    tl.stop()
    ft.assert_true(tl.stopped)
    npt.assert_almost_equal(tl.current_timestamp, 0)

    npt.assert_almost_equal(tl.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(tl.get_position(7), np.array([4, 2, 20]))

    tl.set_position_interpolator(linear_interpolator)
    tl.set_position_interpolator(cubic_bezier_interpolator)
    tl.set_position_interpolator(step_interpolator)
    tl.set_position_interpolator(cubic_spline_interpolator)
    tl.set_position_interpolator(spline_interpolator, degree=2)
    tl.set_rotation_interpolator(step_interpolator)
    tl.set_scale_interpolator(linear_interpolator)
    tl.set_opacity_interpolator(step_interpolator)
    tl.set_color_interpolator(linear_interpolator)

    npt.assert_almost_equal(tl.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(tl.get_position(7), np.array([4, 2, 20]))

    npt.assert_almost_equal(tl.get_color(7), np.array([1, 0, 1]))
    tl.set_color(25, np.array([0.2, 0.2, 0.5]))
    assert_not_equal(tl.get_color(7), np.array([1, 0, 1]))
    assert_not_equal(tl.get_color(25), np.array([0.2, 0.2, 0.5]))

    cube = actor.cube(np.array([[0, 0, 0]]))
    tl.add_actor(cube)

    # using force since the animation is not playing
    tl.update_animation(force=True)

    if not shaders:
        transform = cube.GetUserTransform()
        npt.assert_almost_equal(tl.get_position(tl.current_timestamp),
                                transform.GetPosition())
        npt.assert_almost_equal(tl.get_scale(tl.current_timestamp),
                                transform.GetScale())
        npt.assert_almost_equal(tl.get_rotation(tl.current_timestamp),
                                transform.GetOrientation())
