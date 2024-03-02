import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import transform

from fury.animation.helpers import (
    euclidean_distances,
    get_next_timestamp,
    get_previous_timestamp,
    get_time_tau,
    get_timestamps_from_keyframes,
    get_values_from_keyframes,
    lerp,
)
from fury.colormap import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab, rgb2xyz, xyz2rgb


def spline_interpolator(keyframes, degree):
    """N-th degree spline interpolator for keyframes.

    This is a general n-th degree spline interpolator to be used for any shape
    of keyframes data.

    Parameters
    ----------
    keyframes: dict
        Keyframe data containing timestamps and values to form the spline
        curve. Data should be on the following format:
        >>> {1: {'value': np.array([...])}, 2: {'value': np.array([...])}}

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    """
    if len(keyframes) < (degree + 1):
        raise ValueError(
            f'Minimum {degree + 1} '
            f'keyframes must be set in order to use '
            f'{degree}-degree spline'
        )
    timestamps = get_timestamps_from_keyframes(keyframes)

    values = get_values_from_keyframes(keyframes)
    distances = euclidean_distances(values)
    distances_sum = sum(distances)
    cumulative_dist_sum = np.cumsum([0] + distances)
    tck = splprep(values.T, k=degree, full_output=1, s=0)[0][0]

    def interpolate(t):
        t0 = get_previous_timestamp(timestamps, t)
        t1 = get_next_timestamp(timestamps, t)
        mi_index = np.where(timestamps == t0)[0][0]
        dt = get_time_tau(t, t0, t1)
        section = cumulative_dist_sum[mi_index]
        ts = (section + dt * distances[mi_index]) / distances_sum
        return np.array(splev(ts, tck))

    return interpolate


def cubic_spline_interpolator(keyframes):
    """Cubic spline interpolator for keyframes.

    This is a general cubic spline interpolator to be used for any shape of
    keyframes data.

    Parameters
    ----------
    keyframes: dict
        Keyframe data containing timestamps and values to form the cubic spline
        curve.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    See Also
    --------
    spline_interpolator

    """
    return spline_interpolator(keyframes, degree=3)


def step_interpolator(keyframes):
    """Step interpolator for keyframes.

    This is a simple step interpolator to be used for any shape of
    keyframes data.

    Parameters
    ----------
    keyframes: dict
        Keyframe data containing timestamps and values to form the spline

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)

    def interpolate(t):
        previous_t = get_previous_timestamp(timestamps, t, include_last=True)
        return keyframes.get(previous_t).get('value')

    return interpolate


def linear_interpolator(keyframes):
    """Linear interpolator for keyframes.

    This is a general linear interpolator to be used for any shape of
    keyframes data.

    Parameters
    ----------
    keyframes: dict
        Keyframe data to be linearly interpolated.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)
    is_single = len(keyframes) == 1

    def interpolate(t):
        if is_single:
            t = timestamps[0]
            return keyframes.get(t).get('value')
        t0 = get_previous_timestamp(timestamps, t)
        t1 = get_next_timestamp(timestamps, t)
        p0 = keyframes.get(t0).get('value')
        p1 = keyframes.get(t1).get('value')
        return lerp(p0, p1, t0, t1, t)

    return interpolate


def cubic_bezier_interpolator(keyframes):
    """Cubic Bézier interpolator for keyframes.

    This is a general cubic Bézier interpolator to be used for any shape of
    keyframes data.

    Parameters
    ----------
    keyframes : dict
        Keyframes to be interpolated at any time.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    Notes
    -----
    If no control points are set in the keyframes, The cubic
    Bézier interpolator will almost behave as a linear interpolator.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)

    for ts in timestamps:
        # keyframe at timestamp
        kf_ts = keyframes.get(ts)
        if kf_ts.get('in_cp') is None:
            kf_ts['in_cp'] = kf_ts.get('value')

        if kf_ts.get('out_cp') is None:
            kf_ts['out_cp'] = kf_ts.get('value')

    def interpolate(t):
        t0 = get_previous_timestamp(timestamps, t)
        t1 = get_next_timestamp(timestamps, t)
        k0 = keyframes.get(t0)
        k1 = keyframes.get(t1)
        p0 = k0.get('value')
        p1 = k0.get('out_cp')
        p2 = k1.get('in_cp')
        p3 = k1.get('value')
        dt = get_time_tau(t, t0, t1)
        val = (
            (1 - dt) ** 3 * p0
            + 3 * (1 - dt) ** 2 * dt * p1
            + 3 * (1 - dt) * dt**2 * p2
            + dt**3 * p3
        )
        return val

    return interpolate


def slerp(keyframes):
    """Spherical based rotation keyframes interpolator.

    A rotation interpolator to be used for rotation keyframes.

    Parameters
    ----------
    keyframes : dict
        Rotation keyframes to be interpolated at any time.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    Notes
    -----
    Rotation keyframes must be in the form of quaternions.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)

    quat_rots = []
    for ts in timestamps:
        quat_rots.append(keyframes.get(ts).get('value'))
    rotations = transform.Rotation.from_quat(quat_rots)
    # if only one keyframe specified, linear interpolator is used.
    if len(timestamps) == 1:
        return linear_interpolator(keyframes)
    slerp_interp = transform.Slerp(timestamps, rotations)
    min_t = timestamps[0]
    max_t = timestamps[-1]

    def interpolate(t):
        t = min_t if t < min_t else max_t if t > max_t else t
        v = slerp_interp(t)
        q = v.as_quat()
        return q

    return interpolate


def color_interpolator(keyframes, rgb2space, space2rgb):
    """Custom-space color interpolator.

    Interpolate values linearly inside a custom color space.

    Parameters
    ----------
    keyframes : dict
        Rotation keyframes to be interpolated at any time.
    rgb2space: function
        A functions that take color value in rgb and return that color
         converted to the targeted space.
    space2rgb: function
        A functions that take color value in the targeted space and returns
        that color in rgb space.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)
    space_keyframes = {}
    is_single = len(keyframes) == 1
    for ts, keyframe in keyframes.items():
        space_keyframes[ts] = rgb2space(keyframe.get('value'))

    def interpolate(t):
        if is_single:
            t = timestamps[0]
            return keyframes.get(t).get('value')
        t0 = get_previous_timestamp(timestamps, t)
        t1 = get_next_timestamp(timestamps, t)
        c0 = space_keyframes.get(t0)
        c1 = space_keyframes.get(t1)
        space_color_val = lerp(c0, c1, t0, t1, t)
        return space2rgb(space_color_val)

    return interpolate


def hsv_color_interpolator(keyframes):
    """HSV interpolator for color keyframes

    See Also
    --------
    color_interpolator

    """
    return color_interpolator(keyframes, rgb2hsv, hsv2rgb)


def lab_color_interpolator(keyframes):
    """LAB interpolator for color keyframes

    See Also
    --------
    color_interpolator

    """
    return color_interpolator(keyframes, rgb2lab, lab2rgb)


def xyz_color_interpolator(keyframes):
    """XYZ interpolator for color keyframes

    See Also
    --------
    color_interpolator

    """
    return color_interpolator(keyframes, rgb2xyz, xyz2rgb)


def tan_cubic_spline_interpolator(keyframes):
    """Cubic spline interpolator for keyframes using tangents.
    glTF contains additional tangent information for the cubic spline
    interpolator.

    Parameters
    ----------
    keyframes: dict
        Keyframe data containing timestamps and values to form the cubic spline
        curve.

    Returns
    -------
    function
        The interpolation function that take time and return interpolated
        value at that time.

    """
    timestamps = get_timestamps_from_keyframes(keyframes)
    for time in keyframes:
        data = keyframes.get(time)
        value = data.get('value')
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)

    def interpolate(t):
        t0 = get_previous_timestamp(timestamps, t)
        t1 = get_next_timestamp(timestamps, t)

        dt = get_time_tau(t, t0, t1)

        time_delta = t1 - t0

        p0 = keyframes.get(t0).get('value')
        tan_0 = keyframes.get(t0).get('out_tangent') * time_delta
        p1 = keyframes.get(t1).get('value')
        tan_1 = keyframes.get(t1).get('in_tangent') * time_delta
        # cubic spline equation using tangents
        t2 = dt * dt
        t3 = t2 * dt
        return (
            (2 * t3 - 3 * t2 + 1) * p0
            + (t3 - 2 * t2 + dt) * tan_0
            + (-2 * t3 + 3 * t2) * p1
            + (t3 - t2) * tan_1
        )

    return interpolate
