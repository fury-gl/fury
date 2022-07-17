import math
import time
import numpy as np
from scipy import interpolate
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
        self.max_timestamp = 0
        self.setup()

    def setup(self):
        self.timestamps = np.sort(np.array(list(self.keyframes)), axis=None)
        self.min_timestamp = self.timestamps[0]
        self.max_timestamp = self.timestamps[-1]

    def _get_nearest_smaller_timestamp(self, t, include_last=False):
        if t > self.min_timestamp:
            if include_last:
                return self.timestamps[self.timestamps <= t].max()
            return self.timestamps[:-1][self.timestamps[:-1] <= t].max()
        return self.min_timestamp

    def _get_nearest_larger_timestamp(self, t, include_first=False):
        if t < self.max_timestamp:
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
            k1 = {"t": t_s, "data": self.space_keyframes[t_s]}
            k2 = {"t": t_e, "data": self.space_keyframes[t_e]}
        else:
            k1 = {"t": t_s, "data": self.keyframes[t_s]['value']}
            k2 = {"t": t_e, "data": self.keyframes[t_e]['value']}
        if isinstance(self, CubicBezierInterpolator):
            k1["cp"] = self.keyframes['post_cp'][t_s]
            k2["cp"] = self.keyframes['pre_cp'][t_e]
        return {"start": k1, "end": k2}

    @staticmethod
    def _lerp(v1, v2, t1, t2, t):
        if t1 == t2:
            return v1
        v = v2 - v1
        dt = (t - t1) / (t2 - t1)
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
        return self.keyframes[t_lower]['value']


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

    def setup(self):
        super(LinearInterpolator, self).setup()

    def interpolate(self, t):
        t1 = self._get_nearest_smaller_timestamp(t)
        t2 = self._get_nearest_larger_timestamp(t)
        p1 = self.keyframes[t1]['value']
        p2 = self.keyframes[t2]['value']
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
        points = np.asarray([self.keyframes[i]['value'] for i in
                             self.timestamps])

        if len(points) < (self.degree + 1):
            raise ValueError(f"Minimum {self.degree + 1} "
                             f"keyframes must be set in order to use "
                             f"{self.degree}-degree spline")

        self.tck = interpolate.splprep(points.T, k=self.degree, full_output=1,
                                       s=self.smoothness)[0][0]
        self.linear_lengths = []
        for x, y in zip(points, points[1:]):
            self.linear_lengths.append(
                math.sqrt((x[1] - y[1]) * (x[1] - y[1]) +
                          (x[0] - y[0]) * (x[0] - y[0])))

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
    """

    def __init__(self, keyframes):
        super(CubicBezierInterpolator, self).__init__(keyframes)
        self.id = 2

    def setup(self):
        super(CubicBezierInterpolator, self).setup()
        for ts in self.timestamps:
            if self.keyframes[ts]['pre_cp'] is None:
                self.keyframes[ts]['pre_cp'] = self.keyframes[ts]['value']
            if self.keyframes[ts]['post_cp'] is None:
                self.keyframes[ts]['post_cp'] = self.keyframes[ts]['value']

    def interpolate(self, t):
        t1, t2 = self.get_neighbour_timestamps(t)
        p0 = self.keyframes[t1]['value']
        p1 = self.keyframes[t1]['post_cp']
        p2 = self.keyframes[t2]['pre_cp']
        p3 = self.keyframes[t2]['value']
        if t2 == t1:
            return p0
        dt = (t - t1) / (t2 - t1)
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

    def interpolate(self, t):
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1]
        t = min_t if t < min_t else max_t if t > max_t else t
        return self._slerp(t).as_euler('xyz', degrees=True)


class ColorInterpolator(Interpolator):
    def __init__(self, keyframes, rgb_to_space, space_to_rgb):
        self.rgb_to_space = rgb_to_space
        self.space_to_rgb = space_to_rgb
        self.space_keyframes = {}
        super(ColorInterpolator, self).__init__(keyframes)

    def setup(self):
        super(ColorInterpolator, self).setup()
        for ts, keyframe in self.keyframes.items():
            self.space_keyframes[ts] = self.rgb_to_space(keyframe['value'])

    def interpolate(self, t):
        t1, t2 = self.get_neighbour_timestamps(t)
        p1 = self.space_keyframes[t1]
        p2 = self.space_keyframes[t2]
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
        self.playing = False
        self.loop = False
        self.reversePlaying = False
        self._last_started_at = 0
        self._last_timestamp = 0
        self._current_timestamp = 0
        self.speed = 1
        self._timelines = []
        self._camera = None
        self._scene = None
        self._last_timestamp = 0
        self._last_started_at = 0
        self.playing = False
        self.speed = 2
        self._current_timestamp = 0
        self._has_playback_panel = playback_panel
        self._max_timestamp = 0

        # Handle actors while constructing the timeline.
        if playback_panel:
            self.playback_panel = PlaybackPanel()
            self.playback_panel.on_play_button_clicked = self.play
            self.playback_panel.on_stop_button_clicked = self.stop
            self.playback_panel.on_pause_button_clicked = self.pause
            self.playback_panel.on_progress_bar_changed = self.seek

        if actors is not None:
            self.add_actor(actors)

    def update_max_timestamp(self):
        """Calculate and Get the max timestamp of all keyframes.

        Returns
        -------
        float
            maximum timestamp that can be reached inside the Timeline.
        """

        self._max_timestamp = max(self._max_timestamp,
                                  max([0] + [tl.update_max_timestamp() for tl
                                             in self._timelines]))
        if self._has_playback_panel:
            self.playback_panel.set_max_time(self._max_timestamp)
        return self._max_timestamp

    def set_timestamp(self, timestamp):
        """Set current timestamp of the animation.

        Parameters
        ----------
        timestamp: float
            Current timestamp to be set.
        """
        if self.playing:
            self._last_started_at = time.perf_counter() - timestamp
        else:
            self._last_timestamp = timestamp

    def set_keyframe(self, attrib, timestamp, value, pre_cp=None,
                     post_cp=None, is_camera=False):
        """Set a keyframe for a certain property.

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: float
            Value of the keyframe at the given timestamp.
        is_camera: bool
            Indicated whether setting a camera property or general property.
        pre_cp: float
            The control point in case of using `cubic Bezier interpolator` when
            time exceeds this timestamp.
        post_cp: float
            The control point in case of using `cubic Bezier interpolator` when
            time precedes this timestamp.
        """
        typ = 'attribs'
        if is_camera:
            typ = 'camera'
            self._camera = self._scene.camera()

        if attrib not in self._data['keyframes'][typ]:
            self._data['keyframes'][typ][attrib] = {}
        self._data['keyframes'][typ][attrib][timestamp] = {}
        self._data['keyframes'][typ][attrib][timestamp]['value'] = \
            np.array(value).astype(np.float)
        self._data['keyframes'][typ][attrib][timestamp]['pre_cp'] = pre_cp
        self._data['keyframes'][typ][attrib][timestamp]['post_cp'] = post_cp

        if attrib not in self._data['interpolators'][typ]:
            self._data['interpolators'][typ][attrib] = LinearInterpolator(
                self._data['keyframes'][typ][attrib])

        else:
            self._data['interpolators'][typ][attrib].setup()

        if timestamp > self._max_timestamp:
            self._max_timestamp = timestamp

    def set_keyframes(self, attrib, keyframes, is_camera=False):
        """Set multiple keyframes for a certain property.

        Parameters
        ----------
        attrib: str
            The name of the property.
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
            self.set_keyframe(attrib, t, keyframes[t], is_camera=is_camera)

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

    def set_interpolator(self, attrib, interpolator, is_camera=False):
        """Set keyframes interpolator for a certain property

        Parameters
        ----------
        attrib: str
            The name of the property.
        interpolator: Interpolator
            The interpolator to be used to interpolate keyframes.
        is_camera: bool
            Indicated whether dealing with a camera property or general
            property.

        Examples
        ---------
        >>> Timeline.set_interpolator('position', LinearInterpolator)
        """
        typ = 'attribs'
        if is_camera:
            typ = 'camera'
        if attrib in self._data['keyframes'][typ]:
            self._data['interpolators'][typ][attrib] = interpolator(
                self._data['keyframes'][typ][attrib])

    def is_interpolatable(self, attrib, is_camera=False):
        """Checks if a property is interpolatable.

        Parameters
        ----------
        attrib: str
            The name of the property.
        is_camera: bool
            Indicated whether checking a camera property or general property.

        Returns
        -------
        bool
            Interpolatable state.

        Notes
        -------
        True means that it's safe to use interpolator.interpolate(t) for the
        specified property. And False means the opposite.

        Examples
        ---------
        >>> Timeline.set_interpolator('position', LinearInterpolator)
        """
        typ = 'camera' if is_camera else 'attribs'
        return attrib in self._data['interpolators'][typ]

    def set_camera_interpolator(self, attrib, interpolator):

        self.set_interpolator(attrib, interpolator, is_camera=True)

    def set_position_interpolator(self, interpolator):
        self.set_interpolator('position', interpolator)

    def set_scale_interpolator(self, interpolator):
        self.set_interpolator('scale', interpolator)

    def set_color_interpolator(self, interpolator):
        self.set_interpolator('color', interpolator)

    def set_opacity_interpolator(self, interpolator):
        self.set_interpolator('opacity', interpolator)

    def set_camera_position_interpolator(self, interpolator):
        self.set_camera_interpolator("position", interpolator)

    def set_camera_focal_interpolator(self, interpolator):
        self.set_camera_interpolator("focal", interpolator)

    def get_property_value(self, attrib, t):
        return self._data['interpolators']['attribs'][attrib].interpolate(t)

    def get_camera_property_value(self, attrib, t):
        return self._data['interpolators']['camera'][attrib].interpolate(t)

    def set_position(self, timestamp, position, pre_cp=None, post_cp=None):
        self.set_keyframe('position', timestamp, position, pre_cp, post_cp)

    def set_position_keyframes(self, keyframes):
        self.set_keyframes('position', keyframes)

    def set_rotation(self, timestamp, quat):
        # not functional yet
        pass

    def set_scale(self, timestamp, scalar):
        self.set_keyframe('scale', timestamp, scalar)

    def set_scale_keyframes(self, keyframes):
        self.set_keyframes('scale', keyframes)

    def set_color(self, timestamp, color):
        self.set_keyframe('color', timestamp, color)

    def set_color_keyframes(self, keyframes):
        self.set_keyframes('color', keyframes)

    def set_opacity(self, timestamp, opacity):
        """Value from 0 to 1"""
        self.set_keyframe('opacity', timestamp, opacity)

    def set_opacity_keyframes(self, keyframes):
        self.set_keyframes('opacity', keyframes)

    def get_position(self, t):
        return self.get_property_value('position', t)

    def get_rotation(self, t):
        # not functional yet
        pass

    def get_scale(self, t):
        return self.get_property_value('scale', t)

    def get_color(self, t):
        return self.get_property_value('color', t)

    def get_opacity(self, t):
        return self.get_property_value('opacity', t)

    def set_camera_position(self, timestamp, position):
        self.set_camera_keyframe('position', timestamp, position)

    def set_camera_position_keyframes(self, keyframes):
        self.set_camera_keyframes('position', keyframes)

    def set_camera_focal(self, timestamp, position):
        self.set_camera_keyframe('focal', timestamp, position)

    def set_camera_focal_keyframes(self, keyframes):
        self.set_camera_keyframes('focal', keyframes)

    def set_camera_view_up(self, timestamp, direction):
        self.set_camera_keyframe('view_up', timestamp, direction)

    def set_camera_view_up_keyframes(self, keyframes):
        self.set_camera_keyframes('view_up', keyframes)

    def get_camera_position(self, t):
        return self.get_camera_property_value('position', t)

    def get_camera_focal(self, t):
        return self.get_camera_property_value('focal', t)

    def get_camera_view_up(self, t):
        return self.get_camera_property_value('view_up', t)

    def add(self, item):
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
        self._timelines.append(timeline)

    def add_actor(self, actor):
        if isinstance(actor, list):
            for a in actor:
                self.add_actor(a)
            return
        actor.vcolors = utils.colors_from_actor(actor)
        super(Timeline, self).add(actor)

    def get_actors(self):
        return self.items

    def get_timelines(self):
        return self._timelines

    def remove_timelines(self):
        self._timelines.clear()

    def remove_actor(self, actor):
        self._items.remove(actor)

    def remove_actors(self):
        self.clear()

    def update_animation(self, t=None, force=False):
        """Update the timelines"""
        if t is None:
            t = self.get_current_timestamp()
            if t >= self._max_timestamp:
                self.pause()
        if self._has_playback_panel and not force and t < self._max_timestamp:
            self.playback_panel.set_time(t)
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

            if self.is_interpolatable('color'):
                color = self.get_color(t)
                for act in self.get_actors():
                    act.vcolors[:] = color * 255
                    utils.update_actor(act)

        [tl.update_animation(t, force=True) for tl in self._timelines]

    def play(self):
        """Play the animation"""
        self.update_max_timestamp()
        if not self.playing:
            self._last_started_at = time.perf_counter() - self._last_timestamp
            self.playing = True

    def pause(self):
        """Pause the animation"""
        self._last_timestamp = self.get_current_timestamp()
        self.playing = False

    def stop(self):
        """Stops the animation"""
        self._last_timestamp = 0
        self.playing = False
        self.update_animation(force=True)

    def restart(self):
        """Restarts the animation"""
        self._last_timestamp = 0
        self.playing = True
        self.update_animation(force=True)

    def get_current_timestamp(self):
        """Get current timestamp of the animation"""
        if self.playing:
            return (time.perf_counter() - self._last_started_at) * 1
        else:
            return self._last_timestamp

    def seek(self, t):
        """Change the current timestamp of the animation"""
        if self.playing:
            self._last_started_at = time.perf_counter() - t
        else:
            self._last_timestamp = t
            self.update_animation(force=True)

    def seek_percent(self, p):
        """Change the current timestamp of the animation given a value from
        0 to 100
        """
        t = p * self._max_timestamp / 100
        self.seek(t)

    def is_playing(self):
        """Get the playing status of the timeline"""
        return self.playing

    def is_stopped(self):
        """Get the stopped status of the timeline"""
        return not self.playing and not self._last_timestamp

    def is_paused(self):
        """Get the paused status of the timeline"""
        return not self.playing and self._last_timestamp

    def set_speed(self, speed):
        """Set the speed of the timeline"""
        self.speed = speed

    def add_to_scene(self, ren):
        super(Timeline, self).add_to_scene(ren)
        if self._has_playback_panel:
            ren.add(self.playback_panel)
        [ren.add(timeline) for timeline in self._timelines]
        self._scene = ren
