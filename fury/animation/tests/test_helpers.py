import numpy as np
import numpy.testing as npt
import fury.animation.helpers as helpers
import fury.testing as ft


def test_get_timestamps_from_keyframes():
    keyframes = {
        0: {'value': np.array([0, 0, 0])},
        1: {'value': np.array([1, 0, 0])},
        2: {'value': np.array([2, 0, 0])},
    }
    # Test `get_timestamps_from_keyframes`
    timestamps = helpers.get_timestamps_from_keyframes(keyframes)
    npt.assert_equal(len(timestamps), len(keyframes))
    npt.assert_equal(np.array(sorted(keyframes.keys())), timestamps)


def test_lerp():
    v0 = np.array([0, 0, 0, 0, 0])
    v1 = np.array([1, 1, 1, 1, 1])
    t0 = 1
    t1 = 2
    for t in range(-100, 100, 1):
        t /= 10
        # lerp
        interp_value = helpers.lerp(v0, v1, t0, t1, t)
        npt.assert_array_equal(helpers.get_time_tau(t, t0, t1) * (v1 - v0) +
                               v0, interp_value)
    npt.assert_array_equal(helpers.lerp(v0, v1, t0, t1, t0), v0)
    npt.assert_array_equal(helpers.lerp(v0, v1, t0, t1, t1), v1)


def test_get_values_from_keyframes():
    keyframes = {
        0: {'value': np.array([0, 0, 0])},
        1: {'value': np.array([1, 0, 0])},
        2: {'value': np.array([2, 0, 0])},
    }
    values = helpers.get_values_from_keyframes(keyframes)
    npt.assert_array_equal(values, np.array([i['value'] for i in
                                             keyframes.values()]))

    values = helpers.get_values_from_keyframes({})
    npt.assert_array_equal(values, np.array([]))

    values = helpers.get_values_from_keyframes({1: {}})
    npt.assert_array_equal(values, np.array([None]))


def test_get_next_timestamp():
    timestamps = np.array([1, 2, 3, 4, 5, 6])
    for t in range(-100, 100, 1):
        t /= 10
        next_ts = helpers.get_next_timestamp(timestamps, t)
        npt.assert_(next_ts in timestamps, "Timestamp is not valid")
        ft.assert_greater_equal(next_ts, min(max(timestamps), t))
        next_ts_2 = helpers.get_next_timestamp(timestamps, t,
                                               include_first=True)
        ft.assert_less_equal(next_ts_2, next_ts)
        npt.assert_(next_ts_2 in timestamps, "Timestamp is not valid")

    ts = helpers.get_next_timestamp(timestamps, 0.5, include_first=False)
    ft.assert_equal(ts, 2)
    ts = helpers.get_next_timestamp(timestamps, 0.5, include_first=True)
    ft.assert_equal(ts, 1)


def test_get_previous_timestamp():
    timestamps = np.array([1, 2, 3, 4, 5, 6])
    for t in range(-100, 100, 1):
        t /= 10
        previous_ts = helpers.get_previous_timestamp(timestamps, t)
        npt.assert_(previous_ts in timestamps, "Timestamp is not valid")
        ft.assert_less_equal(previous_ts, max(min(timestamps), t))
        previous_ts_2 = helpers.get_previous_timestamp(timestamps, t,
                                                       include_last=True)
        ft.assert_greater_equal(previous_ts_2, previous_ts)
        npt.assert_(previous_ts_2 in timestamps, "Timestamp is not valid")

    ts = helpers.get_previous_timestamp(timestamps, 5.5, include_last=False)
    ft.assert_equal(ts, 5)
    ts = helpers.get_previous_timestamp(timestamps, 5.5, include_last=True)
    ft.assert_equal(ts, 5)
    ts = helpers.get_previous_timestamp(timestamps, 7, include_last=False)
    ft.assert_equal(ts, 5)
    ts = helpers.get_previous_timestamp(timestamps, 7, include_last=True)
    ft.assert_equal(ts, 6)


def test_get_time_tau():
    t0 = 5
    t1 = 20
    for t in range(-100, 100, 1):
        t /= 10
        tt = helpers.get_time_tau(t, t0, t1)
        ft.assert_greater_equal(tt, 0)
        ft.assert_less_equal(tt, 1)
    ft.assert_equal(helpers.get_time_tau(5, 5, 20), 0)
    ft.assert_equal(helpers.get_time_tau(20, 5, 20), 1)
    ft.assert_equal(helpers.get_time_tau(14, 5, 20), 0.6)
    ft.assert_equal(helpers.get_time_tau(1.5, 1, 2), 0.5)


def test_euclidean_distances():
    points = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([2, 0, 0]),
        np.array([0, 0, 0]),
    ]
    distance = helpers.euclidean_distances(points)
    expected_distances = np.array([1, 1, 2])
    npt.assert_equal(distance, expected_distances)
