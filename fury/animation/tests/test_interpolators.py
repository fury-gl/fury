from itertools import combinations
import numpy as np
import numpy.testing as npt
from fury.animation.interpolator import linear_interpolator, \
    step_interpolator, cubic_spline_interpolator, cubic_bezier_interpolator, \
    spline_interpolator, hsv_color_interpolator, lab_color_interpolator, \
    xyz_color_interpolator, slerp


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

    interp = step_interpolator({})
    try:
        interp(1)
        raise "This shouldn't work since no keyframes were provided!"
    except IndexError:
        ...

    data = {1: {'value': np.array([1, 2, 3])}}
    interp = step_interpolator(data)
    npt.assert_equal(interp(-100), np.array([1, 2, 3]))
    npt.assert_equal(interp(100), np.array([1, 2, 3]))

    data = {1: {'value': None}}
    interp = step_interpolator(data)
    npt.assert_equal(interp(-100), None)


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

    interp = linear_interpolator({})
    try:
        interp(1)
        raise "This shouldn't work since no keyframes were provided!"
    except IndexError:
        ...

    data = {1: {'value': np.array([1, 2, 3])}}
    interp = linear_interpolator(data)
    npt.assert_equal(interp(-100), np.array([1, 2, 3]))
    npt.assert_equal(interp(100), np.array([1, 2, 3]))

    data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])}}
    interp = linear_interpolator(data)
    try:
        interp(1)
        raise "This shouldn't work since invalid keyframes were provided!"
    except TypeError:
        ...


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

    try:
        cubic_spline_interpolator({})
        raise "At least 4 keyframes must be provided!"
    except ValueError:
        ...

    data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])},
            3: {'value': None}, 4: {'value': None}}
    try:
        cubic_spline_interpolator(data)
        raise "Interpolator should not work with invalid data!"
    except TypeError:
        ...


def test_cubic_bezier_interpolator():
    data_1 = {1: {'value': np.array([-2, 0, 0])},
              2: {'value': np.array([18, 0, 0])}}

    data_2 = {
        1: {'value': np.array([-2, 0, 0]),
            'out_cp': np.array([-15, 6, 0])},
        2: {'value': np.array([18, 0, 0]),
            'in_cp': np.array([27, 18, 0])}
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

    interp = cubic_bezier_interpolator({})

    try:
        interp(1)
        raise "This shouldn't work since no keyframes were provided!"
    except IndexError:
        ...

    data = {1: {'value': np.array([1, 2, 3])}}
    interp = cubic_bezier_interpolator(data)
    npt.assert_equal(interp(-10), np.array([1, 2, 3]))
    npt.assert_equal(interp(100), np.array([1, 2, 3]))

    data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])}}
    interp = cubic_bezier_interpolator(data)
    try:
        interp(1)
        raise "This shouldn't work since no keyframes were provided!"
    except TypeError:
        ...


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
    try:
        spline_interpolator({}, 5)
        raise "At least 6 keyframes must be provided!"
    except ValueError:
        ...

    data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])},
            3: {'value': None}, 4: {'value': None}}
    try:
        spline_interpolator(data, 2)
        raise "Interpolator should not work with invalid data!"
    except TypeError:
        ...


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
        color_interps_functions = [
            hsv_color_interpolator,
            linear_interpolator,
            lab_color_interpolator,
            xyz_color_interpolator,
        ]
        for interpolator in color_interps_functions:
            interp = interpolator({})
            try:
                interp(1)
                raise "This shouldn't work since no keyframes were provided!"
            except IndexError:
                ...

            data = {1: {'value': np.array([1, 2, 3])}}
            interp = interpolator(data)
            npt.assert_equal(interp(-10), np.array([1, 2, 3]))
            npt.assert_equal(interp(10), np.array([1, 2, 3]))

            data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])}}
            try:
                interpolator(data)
                raise "This shouldn't work since invalid keyframes " \
                      "were provided! and hence cant be converted to " \
                      "targeted color space."
            except (TypeError, AttributeError,):
                ...


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

    try:
        interp = slerp({})
        interp(1)
        raise "This shouldn't work since no keyframes were provided!"
    except ValueError:
        ...

    data = {1: {'value': np.array([1, 2, 3, 1])}}
    interp = slerp(data)
    npt.assert_equal(interp(-100), np.array([1, 2, 3, 1]))
    npt.assert_equal(interp(100), np.array([1, 2, 3, 1]))

    data = {1: {'value': None}, 2: {'value': np.array([1, 1, 1])}}
    try:
        interp = slerp(data)
        interp(1)
        raise "This shouldn't work since invalid keyframes were provided!"
    except ValueError:
        ...
