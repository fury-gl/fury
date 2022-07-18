import time
import numpy as np
from scipy import interpolate
from scipy.spatial import transform
from fury import utils
from fury.actor import Container
from fury.colormap import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb, xyz2rgb, rgb2xyz
from fury.ui.elements import PlaybackPanel
from vtkmodules.vtkRenderingCore import vtkActor


class Interpolator(object):
    def __init__(self, keyframes):
        super(Interpolator, self).__init__()
        self.keyframes = keyframes
        self.timestamps = []
        self.min_timestamp = 0
        self.final_timestamp = 0
        self._unity_kf = True
        self.setup()

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
    def _lerp(v1, v2, t1, t2, t):
        if t1 == t2:
            return v1
        v = v2 - v1
        dt = 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)
        return dt * v + v1

    @staticmethod
    def _get_time_delta(t, t1, t2):
        return 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)


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
        return self._lerp(p1, p2, t1, t2, t)


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
        dt = self._get_time_delta(t, t1, t2)
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
    """Cubic bezier interpolator for keyframes.

    This is a general cubic bezier interpolator to be used for any shape of
    keyframes data.

    Attributes
    ----------
    keyframes : dict
        Keyframes to be interpolated at any time.

    Notes
    -----
    If no control points are set in the keyframes, The cubic
    Bezier interpolator will almost as the linear interpolator
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
        dt = self._get_time_delta(t, t1, t2)
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
        timestamps, euler_rots = [], []
        for ts in self.keyframes:
            timestamps.append(ts)
            euler_rots.append(self.keyframes.get(ts).get('value'))
        rotations = transform.Rotation.from_euler('xyz', euler_rots,
                                                  degrees=True)
        self._slerp = transform.Slerp(timestamps, rotations)

    @staticmethod
    def _quaternion2euler(x, y, z, w):
        y_2 = y ** 2

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y_2)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = 2.0 * (w * y - z * x)

        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        Y = np.degrees(np.arcsin(t2))

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y_2 + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z

    def interpolate(self, t):
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1]
        t = min_t if t < min_t else max_t if t > max_t else t
        v = self._slerp(t)
        q = v.as_quat()
        return self._quaternion2euler(*q)


class ColorInterpolator(Interpolator):
    """Color keyframes interpolator.

    A color interpolator to be used for color keyframes.
    Given two functions, one to convert from rgb space to the interpolation
    space, the other is to
    Attributes
    ----------
    keyframes : dict
        Keyframes to be interpolated at any time.

    Notes
    -----
    If no control points are set in the keyframes, The cubic
    Bezier interpolator will almost as the linear interpolator
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
        lab_val = self._lerp(p1, p2, t1, t2, t)
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


class Timeline(Container):
    """Keyframe animation timeline class.

    This timeline is responsible for keyframe animations for a single or a
    group of models.
    It's used to handle multiple attributes and properties of Fury actors such
    as transformations, color, and scale.
    It also accepts custom data and interpolates them such as temperature.
    Linear interpolation is used by default to interpolate data between main
    keyframes.
    """

    def __init__(self, actors=None, playback_panel=False):
        super().__init__()
        self._data = {
            'keyframes': {
                'attribs': {},
                'camera': {}
            },
            'interpolators': {
                'attribs': {},
                'camera': {}
            }
        }
        self._last_timestamp = 0
        self._current_timestamp = 0
        self._speed = 1
        self._timelines = []
        self._camera = None
        self._scene = None
        self._last_started_time = 0
        self._playing = False
        self._has_playback_panel = playback_panel
        self._final_timestamp = 0
        self._needs_update = False
        self._reverse_playing = False
        self._loop = False

        # Handle actors while constructing the timeline.
        if playback_panel:
            self.playback_panel = PlaybackPanel()
            self.playback_panel.on_play_button_clicked = self.play
            self.playback_panel.on_stop_button_clicked = self.stop
            self.playback_panel.on_pause_button_clicked = self.pause
            self.playback_panel.on_progress_bar_changed = self.seek

        if actors is not None:
            self.add_actor(actors)

    def update_final_timestamp(self):
        """Calculate and Get the final timestamp of all keyframes.

        Returns
        -------
        float
            final timestamp that can be reached inside the Timeline.
        """

        self._final_timestamp = max(self._final_timestamp,
                                    max([0] + [tl.update_final_timestamp() for
                                               tl in self._timelines]))
        if self._has_playback_panel:
            self.playback_panel.final_time = self._final_timestamp
        return self._final_timestamp

    def set_timestamp(self, timestamp):
        """Set current timestamp of the animation.

        Parameters
        ----------
        timestamp: float
            Current timestamp to be set.
        """
        if self.playing:
            self._last_started_time = \
                time.perf_counter() - timestamp / self.speed
        else:
            self._last_timestamp = timestamp

    def set_keyframe(self, attrib, timestamp, value, pre_cp=None,
                     post_cp=None, is_camera=False):
        """Set a keyframe for a certain attribute.

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: ndarray
            Value of the keyframe at the given timestamp.
        is_camera: bool
            Indicated whether setting a camera property or general property.
        pre_cp: ndarray, shape (1, M), optional
            The control point in case of using `cubic Bezier interpolator` when
            time exceeds this timestamp.
        post_cp: ndarray, shape (1, M), optional
            The control point in case of using `cubic Bezier interpolator` when
            time precedes this timestamp.
        """
        typ = 'attribs'
        if is_camera:
            typ = 'camera'
            self._camera = self._scene.camera()

        keyframes = self._data.get('keyframes')
        if attrib not in keyframes.get(typ):
            keyframes.get(typ)[attrib] = {}
        attrib_keyframes = self._data.get('keyframes').get(typ).get(attrib)
        attrib_keyframes[timestamp] = {
            'value': np.array(value).astype(np.float),
            'pre_cp': pre_cp,
            'post_cp': post_cp
        }
        interpolators = self._data.get('interpolators')
        if attrib not in interpolators.get(typ):
            interpolators.get(typ)[attrib] = \
                LinearInterpolator(attrib_keyframes)

        else:
            interpolators.get(typ).get(attrib).setup()

        if timestamp > self.final_timestamp:
            self._final_timestamp = timestamp
            if self._has_playback_panel:
                final_t = self.update_final_timestamp()
                self.playback_panel.final_time = final_t

    def set_keyframes(self, attrib, keyframes, is_camera=False):
        """Set multiple keyframes for a certain attribute.

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        keyframes: dict
            A dict object containing keyframes to be set.
        is_camera: bool
            Indicated whether setting a camera property or general property.

        Notes
        ---------
        Cubic Bezier control points are not supported yet in this setter.

        Examples
        ---------
        >>> pos_keyframes = {1: np.array([1, 2, 3]), 3: np.array([5, 5, 5])}
        >>> Timeline.set_keyframes('position', pos_keyframes)
        """
        for t in keyframes:
            keyframe = keyframes.get(t)
            self.set_keyframe(attrib, t, keyframe, is_camera=is_camera)

    def set_camera_keyframe(self, attrib, timestamp, value, pre_cp=None,
                            post_cp=None):
        """Set a keyframe for a camera property

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: float
            Value of the keyframe at the given timestamp.
        pre_cp: float
            The control point in case of using `cubic Bezier interpolator` when
            time exceeds this timestamp.
        post_cp: float
            The control point in case of using `cubic Bezier interpolator` when
            time precedes this timestamp.
        """
        self.set_keyframe(attrib, timestamp, value, pre_cp, post_cp, True)

    def set_camera_keyframes(self, attrib, keyframes):
        """Set multiple keyframes for a certain camera property

        Parameters
        ----------
        attrib: str
            The name of the property.
        keyframes: dict
            A dict object containing keyframes to be set.

        Notes
        ---------
        Cubic Bezier control points are not supported yet in this setter.

        Examples
        ---------
        >>> cam_pos = {1: np.array([1, 2, 3]), 3: np.array([5, 5, 5])}
        >>> Timeline.set_camera_keyframes('position', cam_pos)
        """
        self.set_keyframes(attrib, keyframes, is_camera=True)

    def set_interpolator(self, attrib, interpolator, is_camera=False,
                         spline_degree=2):
        """Set keyframes interpolator for a certain property

        Parameters
        ----------
        attrib: str
            The name of the property.
        interpolator: class
            The interpolator to be used to interpolate keyframes.
        is_camera: bool, optional
            Indicated whether dealing with a camera property or general
            property.
        spline_degree: int, optional
            The degree of the spline in case of setting a spline interpolator.

        Examples
        ---------
        >>> Timeline.set_interpolator('position', LinearInterpolator)
        """
        typ = 'attribs'
        if is_camera:
            typ = 'camera'
        if attrib in self._data.get('keyframes').get(typ):
            keyframes = self._data.get('keyframes').get(typ).get(attrib)
            self._data.get('interpolators').get(typ)[attrib] = \
                interpolator(keyframes)

    def is_interpolatable(self, attrib, is_camera=False):
        """Checks whether a property is interpolatable.

        Parameters
        ----------
        attrib: str
            The name of the property.
        is_camera: bool
            Indicated whether checking a camera property or general property.

        Returns
        -------
        bool
            True if the property is interpolatable by the Timeline.

        Notes
        -------
        True means that it's safe to use interpolator.interpolate(t) for the
        specified property. And False means the opposite.

        Examples
        ---------
        >>> Timeline.set_interpolator('position', LinearInterpolator)
        """
        typ = 'camera' if is_camera else 'attribs'
        return attrib in self._data.get('interpolators').get(typ)

    def set_camera_interpolator(self, attrib, interpolator):
        """Set the interpolator for a specific camera property.

        Parameters
        ----------
        attrib: str
            The name of the camera property.
            The already handeled properties are position, focal, and view_up.

        interpolator: class
            The interpolator that handles the camera property interpolation
            between keyframes.

        Examples
        ---------
        >>> Timeline.set_camera_interpolator('focal', LinearInterpolator)
        """
        self.set_interpolator(attrib, interpolator, is_camera=True)

    def set_position_interpolator(self, interpolator):
        """Set the position interpolator for all eactors inside the
        timeline.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the position interpolation between
            keyframes.

        Examples
        ---------
        >>> Timeline.set_position_interpolator(CubicBezierInterpolator)
        """
        self.set_interpolator('position', interpolator)

    def set_scale_interpolator(self, interpolator):
        """Set the scale interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the scale interpolation between
            keyframes.

        Examples
        ---------
        >>> Timeline.set_scale_interpolator(StepInterpolator)
        """
        self.set_interpolator('scale', interpolator)

    def set_rotation_interpolator(self, interpolator):
        """Set the scale interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the rotation (orientation)
            interpolation between keyframes.

        Examples
        ---------
        >>> Timeline.set_rotation_interpolator(Slerp)
        """
        self.set_interpolator('rotation', interpolator)

    def set_color_interpolator(self, interpolator):
        """Set the color interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the color interpolation between
            color keyframes.

        Examples
        ---------
        >>> Timeline.set_color_interpolator(LABInterpolator)
        """
        self.set_interpolator('color', interpolator)

    def set_opacity_interpolator(self, interpolator):
        """Set the opacity interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the opacity interpolation between
            keyframes.

        Examples
        ---------
        >>> Timeline.set_opacity_interpolator(StepInterpolator)
        """
        self.set_interpolator('opacity', interpolator)

    def set_camera_position_interpolator(self, interpolator):
        """Set the camera position interpolator.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the interpolation of camera
            position keyframes.
        """
        self.set_camera_interpolator("position", interpolator)

    def set_camera_focal_interpolator(self, interpolator):
        """Set the camera focal position interpolator.

        Parameters
        ----------
        interpolator: class
            The interpolator class to handle the interpolation of camera
            focal position keyframes.
        """
        self.set_camera_interpolator("focal", interpolator)

    def get_value(self, attrib, timestamp):
        """Returns the value of an attribute at any given timestamp.

        Parameters
        ----------
        attrib: str
            The attribute name.
        timestamp: float
            The timestamp to interpolate at.
        """
        return self._data.get('interpolators').get('attribs').get(
            attrib).interpolate(timestamp)

    def get_camera_value(self, attrib, timestamp):
        """Returns the value of an attribute interpolated at any given
        timestamp.

        Parameters
        ----------
        attrib: str
            The attribute name.
        timestamp: float
            The timestamp to interpolate at.

        """
        return self._data.get('interpolators').get('camera').get(
            attrib).interpolate(timestamp)

    def set_position(self, timestamp, position, pre_cp=None, post_cp=None):
        """Set a position keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        position: ndarray, shape (1, 3)
            Position value
        pre_cp: ndarray, shape (1, 3), optional
            The pre control point for the given position.
        post_cp: ndarray, shape (1, 3), optional
            The post control point for the given position.

        Notes
        -----
        `pre_cp` and `post_cp` only needed when using the cubic bezier
        interpolation method.
        """
        self.set_keyframe('position', timestamp, position, pre_cp, post_cp)

    def set_position_keyframes(self, keyframes):
        """Set a dict of position keyframes at once.
        Should be in the following form:
        {timestamp_1: position_1, timestamp_2: position_2}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 3))
            A dict with timestamps as keys and positions as values.

        Examples
        --------
        >>> pos_keyframes = {1, np.array([0, 0, 0]), 3, np.array([50, 6, 6])}
        >>> Timeline.set_position_keyframes(pos_keyframes)
        """
        self.set_keyframes('position', keyframes)

    def set_rotation(self, timestamp, euler):
        """Set a rotation keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        euler: ndarray, shape(1, 3)
            Euler angles that describes the rotation.
        """
        self.set_keyframe('rotation', timestamp, euler)

    def set_rotation_as_vector(self, timestamp, vector):
        """Set a rotation keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        vector: ndarray, shape(1, 3)
            Directional vector that describes the rotation.
        """
        euler = transform.Rotation.from_rotvec(vector).as_euler('xyz', True)
        self.set_keyframe('rotation', timestamp, euler)

    def set_scale(self, timestamp, scalar):
        """Set a scale keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        vector: ndarray, shape(1, 3)
            Directional vector that describes the rotation.
        """
        self.set_keyframe('scale', timestamp, scalar)

    def set_scale_keyframes(self, keyframes):
        """Set a dict of scale keyframes at once.
        Should be in the following form:
        {timestamp_1: scale_1, timestamp_2: scale_2}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 3))
            A dict with timestamps as keys and scales as values.

        Examples
        --------
        >>> scale_keyframes = {1, np.array([1, 1, 1]), 3, np.array([2, 2, 3])}
        >>> Timeline.set_scale_keyframes(scale_keyframes)
        """
        self.set_keyframes('scale', keyframes)

    def set_color(self, timestamp, color):
        self.set_keyframe('color', timestamp, color)

    def set_color_keyframes(self, keyframes):
        """Set a dict of color keyframes at once.
        Should be in the following form:
        {timestamp_1: color_1, timestamp_2: color_2}

        Parameters
        ----------
        keyframes: dict
            A dict with timestamps as keys and color as values.

        Examples
        --------
        >>> color_keyframes = {1, np.array([1, 0, 1]), 3, np.array([0, 0, 1])}
        >>> Timeline.set_color_keyframes(color_keyframes)
        """
        self.set_keyframes('color', keyframes)

    def set_opacity(self, timestamp, opacity):
        """Value from 0 to 1"""
        self.set_keyframe('opacity', timestamp, opacity)

    def set_opacity_keyframes(self, keyframes):
        """Set a dict of opacity keyframes at once.
        Should be in the following form:
        {timestamp_1: opacity_1, timestamp_2: opacity_2}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 1) or float or int)
            A dict with timestamps as keys and opacities as values.

        Notes
        -----
        Opacity values should be between 0 and 1.

        Examples
        --------
        >>> opacity = {1, np.array([1, 1, 1]), 3, np.array([2, 2, 3])}
        >>> Timeline.set_scale_keyframes(opacity)
        """
        self.set_keyframes('opacity', keyframes)

    def get_position(self, t):
        """Returns the interpolated position.

        Parameters
        ----------
        t: float
            The time to interpolate position at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated position.
        """
        return self.get_value('position', t)

    def get_rotation(self, t):
        """Returns the interpolated rotation.

        Parameters
        ----------
        t: float
            the time to interpolate rotation at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated rotation.
        """
        return self.get_value('rotation', t)

    def get_scale(self, t):
        """Returns the interpolated scale.

        Parameters
        ----------
        t: float
            The time to interpolate scale at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated scale.
        """
        return self.get_value('scale', t)

    def get_color(self, t):
        """Returns the interpolated color.

        Parameters
        ----------
        t: float
            The time to interpolate color value at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated color.
        """
        return self.get_value('color', t)

    def get_opacity(self, t):
        """Returns the opacity value.

        Parameters
        ----------
        t: float
            The time to interpolate opacity at.

        Returns
        -------
        ndarray(1, 1):
            The interpolated opacity.
        """
        return self.get_value('opacity', t)

    def set_camera_position(self, timestamp, position):
        """Returns the camera position.

        Parameters
        ----------
        t: float
            The time to interpolate opacity at.

        Returns
        -------
        ndarray(1, 1):
            The interpolated opacity.
        """
        self.set_camera_keyframe('position', timestamp, position)

    def set_camera_position_keyframes(self, keyframes):
        """Set a dict of camera position keyframes at once.
        Should be in the following form:
        {timestamp_1: position_1, timestamp_2: position_2}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 3))
            A dict with timestamps as keys and opacities as values.

        Examples
        --------
        >>> pos = {0, np.array([1, 1, 1]), 3, np.array([20, 0, 0])}
        >>> Timeline.set_camera_position_keyframes(pos)
        """
        self.set_camera_keyframes('position', keyframes)

    def set_camera_focal(self, timestamp, position):
        self.set_camera_keyframe('focal', timestamp, position)

    def set_camera_focal_keyframes(self, keyframes):
        """Set multiple camera focal position keyframes at once.
        Should be in the following form:
        {timestamp_1: focal_1, timestamp_2: focal_1, ...}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 3))
            A dict with timestamps as keys and camera focal positions as
            values.

        Examples
        --------
        >>> focal_pos = {0, np.array([1, 1, 1]), 3, np.array([20, 0, 0])}
        >>> Timeline.set_camera_focal_keyframes(focal_pos)
        """
        self.set_camera_keyframes('focal', keyframes)

    def set_camera_view_up(self, timestamp, direction):
        self.set_camera_keyframe('view_up', timestamp, direction)

    def set_camera_rotation(self, timestamp, direction):
        self.set_camera_keyframe('rotation', timestamp, direction)

    def set_camera_view_up_keyframes(self, keyframes):
        """Set multiple camera view up direction keyframes.
        Should be in the following form:
        {timestamp_1: view_up_1, timestamp_2: view_up_2, ...}

        Parameters
        ----------
        keyframes: dict(float: ndarray, shape(1, 3))
            A dict with timestamps as keys and camera view up vectors as
            values.

        Examples
        --------
        >>> view_ups = {0, np.array([1, 0, 0]), 3, np.array([0, 1, 0])}
        >>> Timeline.set_camera_view_up_keyframes(pos)
        """
        self.set_camera_keyframes('view_up', keyframes)

    def get_camera_position(self, t):
        """Returns the interpolated camera position.

        Parameters
        ----------
        t: float
            The time to interpolate camera position value at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated camera position.

        Notes
        -----
        The returned position does not necessarily reflect the current camera
        position, but te expected one.
        """
        return self.get_camera_value('position', t)

    def get_camera_focal(self, t):
        """Returns the interpolated camera's focal position.

        Parameters
        ----------
        t: float
            The time to interpolate at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated camera's focal position.

        Notes
        -----
        The returned focal position does not necessarily reflect the current
        camera's focal position, but the expected one.
        """
        return self.get_camera_value('focal', t)

    def get_camera_view_up(self, t):
        """Returns the interpolated camera's view-up directional vector.

        Parameters
        ----------
        t: float
            The time to interpolate at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated camera view-up directional vector.

        Notes
        -----
        The returned focal position does not necessarily reflect the actual
        camera view up directional vector, but the expected one.
        """
        return self.get_camera_value('view_up', t)

    def get_camera_rotation(self, t):
        """Returns the interpolated rotation for the camera expressed
        in euler angles.

        Parameters
        ----------
        t: float
            The time to interpolate at.

        Returns
        -------
        ndarray(1, 3):
            The interpolated camera's rotation.

        Notes
        -----
        The returned focal position does not necessarily reflect the actual
        camera view up directional vector, but the expected one.
        """
        return self.get_camera_value('rotation', t)

    def add(self, item):
        """Adds an item to the Timeline.
        This item can be an actor, Timeline, list of actors, or a list of
        Timelines.

        Parameters
        ----------
        item: Timeline, vtkActor, list(Timeline), or list(vtkActor)
            Actor/s to be animated by the timeline.
        """
        if isinstance(item, list):
            for a in item:
                self.add(a)
            return
        elif isinstance(item, vtkActor):
            self.add_actor(item)
        elif isinstance(item, Timeline):
            self.add_timeline(item)
        else:
            raise ValueError(f"Object of type {type(item)} can't be added to "
                             f"the timeline.")

    def add_timeline(self, timeline):
        """Adds an actor or list of actors to the Timeline.

        Parameters
        ----------
        timeline: Timeline or list(Timeline)
            Actor/s to be animated by the timeline.
        """
        if isinstance(timeline, list):
            for a in timeline:
                self.add_timeline(a)
            return
        self._timelines.append(timeline)

    def add_actor(self, actor):
        """Adds an actor or list of actors to the Timeline.

        Parameters
        ----------
        actor: vtkActor or list(vtkActor)
            Actor/s to be animated by the timeline.
        """
        if isinstance(actor, list):
            for a in actor:
                self.add_actor(a)
            return
        actor.vcolors = utils.colors_from_actor(actor)
        super(Timeline, self).add(actor)

    def get_actors(self):
        """Returns a list of actors.

        Returns
        -------
        list:
            List of actors controlled by the Timeline.
        """
        return self.items

    def get_timelines(self):
        """Returns a list of child Timelines.

        Returns
        -------
        list:
            List of child Timelines of this Timeline.
        """
        return self._timelines

    def remove_timelines(self):
        """Removes all child Timelines from the Timeline"""
        self._timelines.clear()

    def remove_actor(self, actor):
        """Removes an actor from the Timeline.

        Parameters
        ----------
        actor: vtkActor
            Actor to be removed from the timeline.
        """
        self._items.remove(actor)

    def remove_actors(self):
        """Removes all actors from the Timeline"""
        self.clear()

    def update_animation(self, t=None, force=False):
        """Updates the timeline animations"""
        if t is None:
            t = self.current_timestamp
            if t > self._final_timestamp:
                if self._loop:
                    self.seek(0)
                else:
                    self.pause()
        if self._has_playback_panel and not force and \
                t < self._final_timestamp:
            self.playback_panel.current_time = t
        if self.playing or force:
            if self.is_interpolatable('position', is_camera=True):
                cam_pos = self.get_camera_position(t)
                self._camera.SetPosition(cam_pos)

            if self.is_interpolatable('focal', is_camera=True):
                cam_foc = self.get_camera_focal(t)
                self._camera.SetFocalPoint(cam_foc)

            if self.is_interpolatable('view_up', is_camera=True):
                cam_up = self.get_camera_view_up(t)
                self._camera.SetViewUp(cam_up)

            if self.is_interpolatable('rotation', is_camera=True):
                cam_up = self.get_camera_rotation(t)
                self._camera.Set(cam_up)

            if self.is_interpolatable('position'):
                position = self.get_position(t)
                self.SetPosition(position)

            if self.is_interpolatable('scale'):
                scale = self.get_scale(t)
                [act.SetScale(scale) for act in self.get_actors()]

            if self.is_interpolatable('opacity'):
                scale = self.get_opacity(t)
                [act.GetProperty().SetOpacity(scale) for
                 act in self.get_actors()]

            if self.is_interpolatable('rotation'):
                euler = self.get_rotation(t)
                [act.SetOrientation(euler) for
                 act in self.get_actors()]

            if self.is_interpolatable('color'):
                color = self.get_color(t)
                for act in self.get_actors():
                    act.vcolors[:] = color * 255
                    utils.update_actor(act)
        # Also update all child Timelines.
        [tl.update_animation(t, force=True) for tl in self._timelines]

    def play(self):
        """Play the animation"""
        self.update_final_timestamp()
        if not self.playing:
            self._last_started_time = \
                time.perf_counter() - self._last_timestamp / self.speed
            self._playing = True

    def pause(self):
        """Pauses the animation"""
        self._last_timestamp = self.current_timestamp
        self._playing = False

    def stop(self):
        """Stops the animation"""
        self._last_timestamp = 0
        self._playing = False
        self.update_animation(force=True)

    def restart(self):
        """Restarts the animation"""
        self._last_timestamp = 0
        self._playing = True
        self.update_animation(force=True)

    @property
    def current_timestamp(self):
        """Get current timestamp of the Timeline.

        Returns
        ----------
        float
            The current time of the Timeline.

        """
        if self.playing:
            self._last_timestamp = (time.perf_counter() -
                                    self._last_started_time) * self.speed
        return self._last_timestamp

    @current_timestamp.setter
    def current_timestamp(self, timestamp):
        """Set current timestamp of the Timeline.

        Parameters
        ----------
        timestamp: float
            The time to set as current time of the Timeline.

        """
        self.seek(timestamp)

    @property
    def final_timestamp(self):
        """Get the final timestamp of the Timeline.

        Returns
        ----------
        float
            The final time of the Timeline.

        """
        return self._final_timestamp

    def seek(self, timestamp):
        """Sets the current timestamp of the Timeline.

        Parameters
        ----------
        timestamp: float
            The time to seek.

        """
        if self.playing:
            self._last_started_time = \
                time.perf_counter() - timestamp / self.speed
        else:
            self._last_timestamp = timestamp
            self.update_animation(force=True)

    def seek_percent(self, percent):
        """Seek a percentage of the Timeline's final timestamp.

        Parameters
        ----------
        percent: float
            Value from 1 to 100.

        """
        t = percent * self._final_timestamp / 100
        self.seek(t)

    @property
    def playing(self):
        """Return whether the Timeline is playing.

        Returns
        -------
        bool
            Timeline is playing if True.
        """
        return self._playing

    @playing.setter
    def playing(self, playing):
        """Sets the playing state of the Timeline.

        Parameters
        ----------
        playing: bool
            The playing state to be set.

        """
        self._playing = playing

    @property
    def stopped(self):
        """Return whether the Timeline is stopped.

        Returns
        -------
        bool
            Timeline is stopped if True.

        """
        return not self.playing and not self._last_timestamp

    @property
    def paused(self):
        """Return whether the Timeline is paused.

        Returns
        -------
        bool
            Timeline is paused if True.

        """

        return not self.playing and self._last_timestamp is not None

    @property
    def speed(self):
        """Returns the speed of the timeline.

        Returns
        -------
        float
            The speed of the timeline's playback.
        """
        return self._speed

    @speed.setter
    def speed(self, speed):
        """Set the speed of the timeline.

        Parameters
        ----------
        speed: float
            The speed of the timeline's playback.

        """
        self._speed = speed

    def add_to_scene(self, ren):
        super(Timeline, self).add_to_scene(ren)
        if self._has_playback_panel:
            ren.add(self.playback_panel)
        [ren.add(timeline) for timeline in self._timelines]
        self._scene = ren
