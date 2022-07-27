import numpy as np
from scipy import interpolate
from scipy.spatial import transform
from fury.colormap import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, xyz2rgb, rgb2xyz


class Interpolator:
    def __init__(self, keyframes):
        super(Interpolator, self).__init__()
        self.keyframes = keyframes
        self.timestamps = []
        self.min_timestamp = 0
        self.final_timestamp = 0
        self._unity_kf = True
        self.setup()
        self.id = -1

    def setup(self):
        self.timestamps = np.sort(np.array(list(self.keyframes)), axis=None)
        self.min_timestamp = self.timestamps[0]
        self.final_timestamp = self.timestamps[-1]
        if len(self.timestamps) == 1:
            self._unity_kf = True
        else:
            self._unity_kf = False

    def _get_nearest_smaller_timestamp(self, t, include_last=False):
        if t > self.min_timestamp:
            if include_last:
                return self.timestamps[self.timestamps <= t].max()
            return self.timestamps[:-1][self.timestamps[:-1] <= t].max()
        return self.min_timestamp

    def _get_nearest_larger_timestamp(self, t, include_first=False):
        if t < self.final_timestamp:
            if include_first:
                return self.timestamps[self.timestamps > t].min()
            return self.timestamps[1:][self.timestamps[1:] > t].min()
        return self.timestamps[-1]

    def get_neighbour_timestamps(self, t):
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        return t1, t2

    def get_neighbour_keyframes(self, t):
        t_s, t_e = self.get_neighbour_timestamps(t)
        if isinstance(self, ColorInterpolator):
            k1 = {"t": t_s, "data": self.space_keyframes.get(t_s)}
            k2 = {"t": t_e, "data": self.space_keyframes.get(t_e)}
        else:
            k1 = {"t": t_s, "data": self.keyframes.get(t_s).get('value')}
            k2 = {"t": t_e, "data": self.keyframes.get(t_e).get('value')}

        if isinstance(self, CubicBezierInterpolator):
            k1["cp"] = self.keyframes.get('post_cp').get(t_s)
            k2["cp"] = self.keyframes.get('pre_cp').get(t_e)
        return {"start": k1, "end": k2}

    @staticmethod
    def lerp(v1, v2, t1, t2, t):
        if t1 == t2:
            return v1
        v = v2 - v1
        dt = 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)
        return dt * v + v1

    @staticmethod
    def get_time_tau(t, t1, t2):
        return 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)

    @property
    def id(self):
        return self.id

    @id.setter
    def id(self, id):
        self._id = id


class StepInterpolator(Interpolator):
    """Step interpolator for keyframes.

    This is a simple step interpolator to be used for any shape of
    keyframes data.
    """

    def __init__(self, keyframes):
        super(StepInterpolator, self).__init__(keyframes)
        self.id = 0

    def setup(self):
        super(StepInterpolator, self).setup()

    def interpolate(self, t):
        t_lower = self._get_nearest_smaller_timestamp(t, include_last=True)
        return self.keyframes.get(t_lower).get('value')


class LinearInterpolator(Interpolator):
    """Linear interpolator for keyframes.

    This is a general linear interpolator to be used for any shape of
    keyframes data.
    """

    def __init__(self, keyframes=None):
        if keyframes is None:
            keyframes = {}
        super(LinearInterpolator, self).__init__(keyframes)
        self.id = 1

    def interpolate(self, t):
        if self._unity_kf:
            t = self.timestamps[0]
            return self.keyframes.get(t).get('value')
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        p1 = self.keyframes.get(t1).get('value')
        p2 = self.keyframes.get(t2).get('value')
        return self.lerp(p1, p2, t1, t2, t)


class SplineInterpolator(Interpolator):
    """N-th degree spline interpolator for keyframes.

    This is a general n-th degree spline interpolator to be used for any shape
    of keyframes data.
    """

    def __init__(self, keyframes, degree=3, smoothness=3):
        self.degree = degree
        self.smoothness = smoothness
        self.tck = []
        self.linear_lengths = []
        super(SplineInterpolator, self).__init__(keyframes)
        self.id = 6

    def setup(self):
        super(SplineInterpolator, self).setup()
        points = np.asarray([self.keyframes.get(t).get('value') for t in
                             self.timestamps])

        if len(points) < (self.degree + 1):
            raise ValueError(f"Minimum {self.degree + 1} "
                             f"keyframes must be set in order to use "
                             f"{self.degree}-degree spline")

        self.tck = interpolate.splprep(points.T, k=self.degree, full_output=1,
                                       s=self.smoothness)[0][0]
        self.linear_lengths = []
        for x, y in zip(points, points[1:]):
            self.linear_lengths.append(np.linalg.norm(x - y))

    def interpolate(self, t):

        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)

        mi_index = np.where(self.timestamps == t1)[0][0]
        dt = self.get_time_tau(t, t1, t2)
        sect = sum(self.linear_lengths[:mi_index])
        ts = (sect + dt * (self.linear_lengths[mi_index])) / sum(
            self.linear_lengths)
        return np.array(interpolate.splev(ts, self.tck))


class CubicSplineInterpolator(SplineInterpolator):
    """Cubic spline interpolator for keyframes.

    This is a general cubic spline interpolator to be used for any shape of
    keyframes data.
    """

    def __init__(self, keyframes, smoothness=3):
        super(CubicSplineInterpolator, self).__init__(keyframes, degree=3,
                                                      smoothness=smoothness)
        self.id = 7


class CubicBezierInterpolator(Interpolator):
    """Cubic Bézier interpolator for keyframes.

    This is a general cubic Bézier interpolator to be used for any shape of
    keyframes data.

    Attributes
    ----------
    keyframes : dict
        Keyframes to be interpolated at any time.

    Notes
    -----
    If no control points are set in the keyframes, The cubic
    Bézier interpolator will almost behave as a linear interpolator.
    """

    def __init__(self, keyframes):
        super(CubicBezierInterpolator, self).__init__(keyframes)
        self.id = 2

    def setup(self):
        super(CubicBezierInterpolator, self).setup()
        for ts in self.timestamps:
            # keyframe at timestamp
            kf_ts = self.keyframes.get(ts)
            if 'pre_cp' not in kf_ts or kf_ts.get('pre_cp') is None:
                kf_ts['pre_cp'] = kf_ts.get('value')
            else:
                kf_ts['pre_cp'] = np.array(kf_ts.get('pre_cp'))

            if 'post_cp' not in kf_ts or kf_ts.get('post_cp') is None:
                kf_ts['post_cp'] = kf_ts.get('value')
            else:
                kf_ts['post_cp'] = np.array(kf_ts.get('post_cp'))

            # TODO: make it an option to deduce the control point if the
            #  other control point exists

    def interpolate(self, t):
        t1, t2 = self.get_neighbour_timestamps(t)
        p0 = self.keyframes.get(t1).get('value')
        p1 = self.keyframes.get(t1).get('post_cp')
        p2 = self.keyframes.get(t2).get('pre_cp')
        p3 = self.keyframes.get(t2).get('value')
        dt = self.get_time_tau(t, t1, t2)
        res = (1 - dt) ** 3 * p0 + 3 * (1 - dt) ** 2 * dt * p1 + 3 * \
              (1 - dt) * dt ** 2 * p2 + dt ** 3 * p3
        return res


class Slerp(Interpolator):
    """Spherical based rotation keyframes interpolator.

    A rotation interpolator to be used for rotation keyframes.

    Attributes
    ----------
    keyframes : dict
        Rotation keyframes to be interpolated at any time.

    Notes
    -----
    Rotation keyframes must be in the form of Euler degrees.

    """

    def __init__(self, keyframes):
        self._slerp = None
        super(Slerp, self).__init__(keyframes)

    def setup(self):
        super(Slerp, self).setup()
        timestamps, quat_rots = [], []
        for ts in self.keyframes:
            timestamps.append(ts)
            quat_rots.append(self.keyframes.get(ts).get('value'))
        rotations = transform.Rotation.from_quat(quat_rots)
        self._slerp = transform.Slerp(timestamps, rotations)

    def interpolate(self, t):
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1]
        t = min_t if t < min_t else max_t if t > max_t else t
        v = self._slerp(t)
        q = v.as_quat()
        return q


class ColorInterpolator(Interpolator):
    """Color keyframes interpolator.

    A color interpolator to be used for color keyframes.
    Given two functions, one is to convert from RGB space to the interpolation
    space, the other is to convert from that space back to the RGB space.

    Attributes
    ----------
    keyframes : dict
        Keyframes to be interpolated at any time.

    Notes
    -----
    If no control points are set in the keyframes, The cubic Bézier
    interpolator will almost behave as a linear interpolator.
    """

    def __init__(self, keyframes, rgb_to_space, space_to_rgb):
        self.rgb_to_space = rgb_to_space
        self.space_to_rgb = space_to_rgb
        self.space_keyframes = {}
        super(ColorInterpolator, self).__init__(keyframes)

    def setup(self):
        super(ColorInterpolator, self).setup()
        for ts, keyframe in self.keyframes.items():
            self.space_keyframes[ts] = self.rgb_to_space(keyframe.get('value'))

    def interpolate(self, t):
        t1, t2 = self.get_neighbour_timestamps(t)
        p1 = self.space_keyframes.get(t1)
        p2 = self.space_keyframes.get(t2)
        lab_val = self.lerp(p1, p2, t1, t2, t)
        return self.space_to_rgb(lab_val)


class HSVInterpolator(ColorInterpolator):
    """LAB interpolator for color keyframes """

    def __init__(self, keyframes):
        super().__init__(keyframes, rgb2hsv, hsv2rgb)
        self.id = 3


class XYZInterpolator(ColorInterpolator):
    """XYZ interpolator for color keyframes """

    def __init__(self, keyframes):
        super().__init__(keyframes, rgb2xyz, xyz2rgb)
        self.id = 4


class LABInterpolator(ColorInterpolator):
    """LAB interpolator for color keyframes """

    def __init__(self, keyframes):
        super().__init__(keyframes, rgb2lab, lab2rgb)
        self.id = 5
