import time
from itertools import combinations

import numpy as np
import numpy.testing as npt

from fury import actor
import fury.animation.helpers as helpers
from fury.animation.interpolator import linear_interpolator, \
    step_interpolator, cubic_spline_interpolator, cubic_bezier_interpolator, \
    spline_interpolator, hsv_color_interpolator, lab_color_interpolator, \
    xyz_color_interpolator, slerp
from fury.animation.timeline import Timeline
from fury.testing import *
from fury.ui import PlaybackPanel


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_step_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])}}

    interpolator = step_interpolator(data)

    pos1 = interpolator(2)
    pos2 = interpolator(2.9)
    npt.assert_equal(pos1, pos2)

    pos3 = interpolator(3)
    assert_not_equal(pos3, pos2)

    pos_initial = interpolator(1)
    pos_final = interpolator(3)

    # test when time exceeds or precedes the interpolation range
    npt.assert_equal(interpolator(-999), pos_initial)
    npt.assert_equal(interpolator(999), pos_final)

    for t in range(-10, 40, 1):
        npt.assert_equal(interpolator(t / 10).shape,
                         data.get(1).get('value').shape)

    for ts, pos in data.items():
        npt.assert_equal(interpolator(ts),
                         data.get(ts).get('value'))


def test_linear_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])}}

    interpolator = linear_interpolator(data)

    pos1 = interpolator(2)
    pos2 = interpolator(2.1)
    assert_not_equal(pos1, pos2)

    npt.assert_equal(pos1, data.get(2).get('value'))

    for ts, pos in data.items():
        npt.assert_equal(interpolator(ts),
                         data.get(ts).get('value'))

    for t in range(-10, 40, 1):
        npt.assert_equal(interpolator(t / 10).shape,
                         data.get(1).get('value').shape)

    pos_initial = interpolator(1)
    pos_final = interpolator(3)

    # test when time exceeds or precedes the interpolation range
    npt.assert_equal(interpolator(-999), pos_initial)
    npt.assert_equal(interpolator(999), pos_final)


def test_cubic_spline_interpolator():
    data = {1: {'value': np.array([1, 2, 3])},
            2: {'value': np.array([0, 0, 0])},
            3: {'value': np.array([5, 5, 5])},
            4: {'value': np.array([7, 7, 7])}}

    interpolator = cubic_spline_interpolator(data)

    pos1 = interpolator(2)
    npt.assert_almost_equal(pos1, data.get(2).get('value'))

    for ts, pos in data.items():
        npt.assert_almost_equal(interpolator(ts),
                                data.get(ts).get('value'))

    for t in range(-10, 40, 1):
        npt.assert_almost_equal(interpolator(t / 10).shape,
                                data.get(1).get('value').shape)

    pos_initial = interpolator(1)
    pos_final = interpolator(4)

    # test when time exceeds or precedes the interpolation range
    npt.assert_almost_equal(interpolator(-999), pos_initial)
    npt.assert_almost_equal(interpolator(999), pos_final)


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
    interp_1 = cubic_bezier_interpolator(data_1)
    # without control points
    interp_2 = cubic_bezier_interpolator(data_2)
    # linear interpolator
    interp_linear = linear_interpolator(data_2)

    assert_not_equal(interp_1(1.5), interp_2(1.5))

    npt.assert_equal(interp_1(1.5), interp_linear(1.5))
    assert_not_equal(interp_1(1.2), interp_linear(1.2))
    assert_not_equal(interp_2(1.5), interp_linear(1.5))

    # start and end points
    npt.assert_equal(interp_1(1), interp_2(1))
    npt.assert_equal(interp_1(2), interp_2(2))

    for ts, pos in data_1.items():
        expected = data_1.get(ts).get('value')
        npt.assert_almost_equal(interp_1(ts), expected)
        npt.assert_almost_equal(interp_2(ts), expected)

    for t in range(-10, 40, 1):
        npt.assert_almost_equal(interp_1(t / 10).shape,
                                data_1.get(1).get('value').shape)

    pos_initial = interp_1(1)
    pos_final = interp_2(2)

    # test when time exceeds or precedes the interpolation range
    npt.assert_almost_equal(interp_1(-999), pos_initial)
    npt.assert_almost_equal(interp_2(-999), pos_initial)

    npt.assert_almost_equal(interp_1(999), pos_final)
    npt.assert_almost_equal(interp_2(999), pos_final)


def test_n_spline_interpolator():
    data = {i: {'value': np.random.random(3) * 10} for i in range(10)}

    interps = [spline_interpolator(data, degree=i) for i in range(1, 6)]

    for i in interps:
        npt.assert_equal(i(-999), i(0))
        npt.assert_equal(i(999), i(10))
        for t in range(10):
            npt.assert_almost_equal(i(t), data.get(t).get('value'))
        for t in range(-100, 100, 1):
            i(t / 10)


def test_color_interpolators():
    data = {1: {'value': np.array([1, 0.5, 0])},
            2: {'value': np.array([0.5, 0, 1])}}

    color_interps = [
        hsv_color_interpolator(data),
        linear_interpolator(data),
        lab_color_interpolator(data),
        xyz_color_interpolator(data),
    ]

    for interp in color_interps:
        npt.assert_almost_equal(interp(-999),
                                interp(1))
        npt.assert_almost_equal(interp(999), interp(2))

    for interps in combinations(color_interps, 2):
        for timestamp in data.keys():
            npt.assert_almost_equal(interps[0](timestamp),
                                    interps[1](timestamp))
        # excluded main keyframes
        for timestamp in range(101, 200, 1):
            timestamp /= 100
            assert_not_equal(interps[0](timestamp),
                             interps[1](timestamp))


def test_slerp():
    data = {1: {'value': np.array([0, 0, 0, 1])},
            2: {'value': np.array([0, 0.7071068, 0, 0.7071068])}}

    interp_slerp = slerp(data)
    interp_lerp = linear_interpolator(data)

    npt.assert_equal(interp_slerp(-999),
                     interp_slerp(1))
    npt.assert_equal(interp_slerp(999),
                     interp_slerp(2))

    npt.assert_almost_equal(interp_slerp(1),
                            interp_lerp(1))
    npt.assert_almost_equal(interp_slerp(2),
                            interp_lerp(2))
    assert_not_equal(interp_slerp(1.5),
                     interp_lerp(1.5))

    for timestamp in range(-100, 100, 1):
        timestamp /= 10
        interp_slerp(timestamp)


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
        tl.seek(t)
        assert_less_equal(tl.current_timestamp, tl.final_timestamp)
        assert_greater_equal(tl.current_timestamp, 0)

        assert_greater_equal(tl.current_timestamp,
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
    assert_true(tl.playing)

    tl.pause()
    t_before = tl.current_timestamp
    assert_true(tl.paused)
    time.sleep(0.1)
    npt.assert_almost_equal(tl.current_timestamp, t_before)

    tl.stop()
    assert_true(tl.stopped)
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
        npt.assert_almost_equal(tl.get_position(tl.current_timestamp),
                                cube.GetPosition())
        npt.assert_almost_equal(tl.get_scale(tl.current_timestamp),
                                cube.GetScale())
        npt.assert_almost_equal(tl.get_rotation(tl.current_timestamp),
                                cube.GetOrientation())


def test_helper_functions():
    keyframes = {
        0: {'value': np.array([0, 0, 0])},
        1: {'value': np.array([1, 0, 0])},
        2: {'value': np.array([2, 0, 0])},
    }
    # Test `get_timestamps_from_keyframes`
    timestamps = helpers.get_timestamps_from_keyframes(keyframes)
    npt.assert_equal(len(timestamps), len(keyframes))
    npt.assert_equal(np.array(sorted(keyframes.keys())), timestamps)

    values = helpers.get_values_from_keyframes(keyframes)
    npt.assert_array_equal(values, np.array([i['value'] for i in
                                             keyframes.values()]))
    distance = helpers.get_distances(values)
    expected_distances = np.array([1, 1])
    npt.assert_equal(distance, expected_distances)

    for t in range(-100, 100, 1):

        next_ts = helpers.get_next_timestamp(timestamps, t / 10)
        npt.assert_(next_ts in timestamps, "Timestamp is not valid")
        next_ts_2 = helpers.get_next_timestamp(timestamps, t / 10,
                                               include_first=True)
        npt.assert_(next_ts_2 in timestamps, "Timestamp is not valid")

        prev_ts = helpers.get_previous_timestamp(timestamps, t / 10)
        npt.assert_(prev_ts in timestamps, "Timestamp is not valid")
        prev_ts_2 = helpers.get_previous_timestamp(timestamps, t / 10,
                                                   include_last=True)
        npt.assert_(prev_ts_2 in timestamps, "Timestamp is not valid")

        assert_greater(next_ts, prev_ts)
        assert_greater_equal(next_ts, prev_ts_2)
        assert_greater_equal(next_ts_2, prev_ts_2)
        assert_greater_equal(next_ts, prev_ts_2)
        assert_not_equal(next_ts, prev_ts)

        for i in range(-100, 100, 1):
            i /= 10
            tt = helpers.get_time_tau(i, prev_ts, next_ts)
            assert_greater_equal(tt, 0)
            assert_less_equal(tt, 1)

            # lerp
            v1 = keyframes.get(prev_ts).get('value')
            v2 = keyframes.get(next_ts).get('value')

            interp_value = helpers.lerp(v1, v2, prev_ts, next_ts, i)
            assert_arrays_equal(tt * (v2-v1) + v1, interp_value)
