import time

import numpy as np
import numpy.testing as npt

from fury import actor
from fury.animation import Timeline, LinearInterpolator, StepInterpolator, \
    CubicSplineInterpolator, CubicBezierInterpolator, SplineInterpolator, \
    HSVInterpolator, LABInterpolator, XYZInterpolator, Slerp
from itertools import combinations
from fury.testing import *
from fury.ui import PlaybackPanel


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_step_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])}}

    interpolator = StepInterpolator(data)

    pos1 = interpolator.interpolate(2)
    pos2 = interpolator.interpolate(2.9)
    npt.assert_equal(pos1, pos2)

    pos3 = interpolator.interpolate(3)
    assert_not_equal(pos3, pos2)

    pos_initial = interpolator.interpolate(1)
    pos_final = interpolator.interpolate(3)

    # test when time exceeds or precedes the interpolation range
    npt.assert_equal(interpolator.interpolate(-999), pos_initial)
    npt.assert_equal(interpolator.interpolate(999), pos_final)

    for t in range(-10, 40, 1):
        npt.assert_equal(interpolator.interpolate(t / 10).shape,
                         data.get(1).get('value').shape)

    for ts, pos in data.items():
        npt.assert_equal(interpolator.interpolate(ts),
                         data.get(ts).get('value'))


def test_linear_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])}}

    interpolator = LinearInterpolator(data)

    pos1 = interpolator.interpolate(2)
    pos2 = interpolator.interpolate(2.1)
    assert_not_equal(pos1, pos2)

    npt.assert_equal(pos1, data.get(2).get('value'))

    for ts, pos in data.items():
        npt.assert_equal(interpolator.interpolate(ts),
                         data.get(ts).get('value'))

    for t in range(-10, 40, 1):
        npt.assert_equal(interpolator.interpolate(t / 10).shape,
                         data.get(1).get('value').shape)

    pos_initial = interpolator.interpolate(1)
    pos_final = interpolator.interpolate(3)

    # test when time exceeds or precedes the interpolation range
    npt.assert_equal(interpolator.interpolate(-999), pos_initial)
    npt.assert_equal(interpolator.interpolate(999), pos_final)


def test_cubic_spline_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])},
            4: {'value': np.array([7, 7, 7])}}

    interpolator = CubicSplineInterpolator(data)

    pos1 = interpolator.interpolate(2)
    npt.assert_almost_equal(pos1, data.get(2).get('value'))

    for ts, pos in data.items():
        npt.assert_almost_equal(interpolator.interpolate(ts),
                                data.get(ts).get('value'))

    for t in range(-10, 40, 1):
        npt.assert_almost_equal(interpolator.interpolate(t / 10).shape,
                                data.get(1).get('value').shape)

    pos_initial = interpolator.interpolate(1)
    pos_final = interpolator.interpolate(4)

    # test when time exceeds or precedes the interpolation range
    npt.assert_almost_equal(interpolator.interpolate(-999), pos_initial)
    npt.assert_almost_equal(interpolator.interpolate(999), pos_final)


def test_cubic_bezier_interpolator():
    data_1 = {1: {'value': np.array([-2, 0, 0])},
              2: {'value': np.array([18, 0, 0])}}

    data_2 = {
        1: {'value': np.array([-2, 0, 0]),
            'post_cp': np.array([-15, 6, 0])},
        2: {'value': np.array([18, 0, 0]),
            'pre_cp': np.array([27, 18, 0])}
    }

    # with control points
    interp_1 = CubicBezierInterpolator(data_1)
    # without control points
    interp_2 = CubicBezierInterpolator(data_2)
    # linear interpolator
    interp_linear = LinearInterpolator(data_2)

    assert_not_equal(interp_1.interpolate(1.5), interp_2.interpolate(1.5))

    npt.assert_equal(interp_1.interpolate(1.5), interp_linear.interpolate(1.5))
    assert_not_equal(interp_1.interpolate(1.2), interp_linear.interpolate(1.2))
    assert_not_equal(interp_2.interpolate(1.5), interp_linear.interpolate(1.5))

    # start and end points
    npt.assert_equal(interp_1.interpolate(1), interp_2.interpolate(1))
    npt.assert_equal(interp_1.interpolate(2), interp_2.interpolate(2))

    for ts, pos in data_1.items():
        expected = data_1.get(ts).get('value')
        npt.assert_almost_equal(interp_1.interpolate(ts), expected)
        npt.assert_almost_equal(interp_2.interpolate(ts), expected)

    for t in range(-10, 40, 1):
        npt.assert_almost_equal(interp_1.interpolate(t / 10).shape,
                                data_1.get(1).get('value').shape)

    pos_initial = interp_1.interpolate(1)
    pos_final = interp_2.interpolate(2)

    # test when time exceeds or precedes the interpolation range
    npt.assert_almost_equal(interp_1.interpolate(-999), pos_initial)
    npt.assert_almost_equal(interp_2.interpolate(-999), pos_initial)

    npt.assert_almost_equal(interp_1.interpolate(999), pos_final)
    npt.assert_almost_equal(interp_2.interpolate(999), pos_final)


def test_n_spline_interpolator():
    data = {i: {'value': np.random.random(3) * 10} for i in range(10)}

    interps = [SplineInterpolator(data, degree=i) for i in range(1, 6)]

    for i in interps:
        npt.assert_equal(i.interpolate(-999), i.interpolate(0))
        npt.assert_equal(i.interpolate(999), i.interpolate(10))
        for t in range(-100, 100, 1):
            i.interpolate(t / 10)


def test_color_interpolators():
    data = {1: {'value': np.array([1, 0.5, 0])},
            2: {'value': np.array([0.5, 0, 1])}}

    color_interps = [
        HSVInterpolator(data),
        LinearInterpolator(data),
        LABInterpolator(data),
        XYZInterpolator(data),
    ]

    for interp in color_interps:
        npt.assert_equal(interp.interpolate(-999), interp.interpolate(1))
        npt.assert_equal(interp.interpolate(999), interp.interpolate(2))

    for interps in combinations(color_interps, 2):
        for timestamp in data.keys():
            npt.assert_almost_equal(interps[0].interpolate(timestamp),
                                    interps[1].interpolate(timestamp))
        # excluded main keyframes
        for timestamp in range(101, 200, 1):
            timestamp /= 100
            assert_not_equal(interps[0].interpolate(timestamp),
                             interps[1].interpolate(timestamp))


def test_slerp():
    data = {1: {'value': np.array([90, 0, 0])},
            2: {'value': np.array([0, 0, 180])}}

    interp_slerp = Slerp(data)
    interp_lerp = LinearInterpolator(data)

    npt.assert_equal(interp_slerp.interpolate(-999),
                     interp_slerp.interpolate(1))
    npt.assert_equal(interp_slerp.interpolate(999),
                     interp_slerp.interpolate(2))

    npt.assert_almost_equal(interp_slerp.interpolate(1),
                            interp_lerp.interpolate(1))
    npt.assert_almost_equal(interp_slerp.interpolate(2),
                            interp_lerp.interpolate(2))
    assert_not_equal(interp_slerp.interpolate(1.5),
                     interp_lerp.interpolate(1.5))

    for timestamp in range(-100, 100, 1):
        timestamp /= 10
        interp_slerp.interpolate(timestamp)


def test_timeline():
    shaders = False
    tl = Timeline(playback_panel=Timeline)
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
    assert_true(isinstance(tl.playback_panel, PlaybackPanel))

    for t in [-10, 0, 2.2, 7, 100]:
        tl.playback_panel.current_time = t
        assert_equal(tl.current_timestamp, tl.playback_panel.current_time)

        if 0 <= t <= tl.final_timestamp:
            assert_equal(tl.current_timestamp, t)

    tl.play()
    t_before = tl.current_timestamp
    time.sleep(0.1)
    assert_not_equal(tl.current_timestamp, t_before)
    assert_true(tl.playing)

    tl.pause()
    t_before = tl.current_timestamp
    assert_true(tl.paused)
    time.sleep(0.1)
    assert_equal(tl.current_timestamp, t_before)

    tl.stop()
    assert_true(tl.stopped)
    assert_equal(tl.current_timestamp, 0)

    npt.assert_equal(tl.get_position(0), np.array([0, 0, 0]))
    npt.assert_equal(tl.get_position(7), np.array([4, 2, 20]))

    tl.set_position_interpolator(LinearInterpolator)
    tl.set_position_interpolator(CubicBezierInterpolator)
    tl.set_position_interpolator(StepInterpolator)
    tl.set_position_interpolator(CubicSplineInterpolator)
    tl.set_position_interpolator(SplineInterpolator)
    tl.set_rotation_interpolator(StepInterpolator)
    tl.set_scale_interpolator(LinearInterpolator)
    tl.set_opacity_interpolator(StepInterpolator)
    tl.set_color_interpolator(LinearInterpolator)

    npt.assert_almost_equal(tl.get_position(0), np.array([0, 0, 0]))
    npt.assert_almost_equal(tl.get_position(7), np.array([4, 2, 20]))

    npt.assert_equal(tl.get_color(7), np.array([1, 0, 1]))
    tl.set_color(25, np.array([0.2, 0.2, 0.5]))
    assert_not_equal(tl.get_color(7), np.array([1, 0, 1]))
    assert_not_equal(tl.get_color(25), np.array([0.2, 0.2, 0.5]))

    cube = actor.cube(np.array([[0, 0, 0]]))
    tl.add_actor(cube)

    tl.update_animation()

    if not shaders:
        npt.assert_equal(tl.get_position(tl.current_timestamp),
                         cube.GetPosition())
        npt.assert_equal(tl.get_scale(tl.current_timestamp),
                         cube.GetScale())
        npt.assert_equal(tl.get_rotation(tl.current_timestamp),
                         cube.GetOrientation())


