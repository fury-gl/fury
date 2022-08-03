import numpy as np
import numpy.testing as npt
import fury.animation.helpers as helpers
import fury.testing as ft


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
    distance = helpers.euclidean_distances(values)
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

        ft.assert_greater(next_ts, prev_ts)
        ft.assert_greater_equal(next_ts, prev_ts_2)
        ft.assert_greater_equal(next_ts_2, prev_ts_2)
        ft.assert_greater_equal(next_ts, prev_ts_2)
        ft.assert_not_equal(next_ts, prev_ts)

        for i in range(-100, 100, 1):
            i /= 10
            tt = helpers.get_time_tau(i, prev_ts, next_ts)
            ft.assert_greater_equal(tt, 0)
            ft.assert_less_equal(tt, 1)

            # lerp
            v1 = keyframes.get(prev_ts).get('value')
            v2 = keyframes.get(next_ts).get('value')

            interp_value = helpers.lerp(v1, v2, prev_ts, next_ts, i)
            ft.assert_arrays_equal(tt * (v2-v1) + v1, interp_value)
