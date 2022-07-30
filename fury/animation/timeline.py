import time
import warnings
import inspect
from collections import defaultdict

import numpy as np
from scipy.spatial import transform
from fury import utils, actor
from fury.actor import Container
from fury.animation.interpolator import linear_interpolator
from fury.ui.elements import PlaybackPanel
from fury.lib import Actor


class Timeline(Container):
    """Keyframe animation timeline class.

    This timeline is responsible for keyframe animations for a single or a
    group of models.
    It's used to handle multiple attributes and properties of Fury actors such
    as transformations, color, and scale.
    It also accepts custom data and interpolates them, such as temperature.
    Linear interpolation is used by default to interpolate data between the
    main keyframes.

    Attributes
    ----------
    actors : str
        a formatted string to print out what the animal says
    playback_panel : bool, optional
        If True, the timeline will have a playback panel set, which can be used
        to control the playback of the timeline.
    length : float or int, default: None, optional
        the fixed length of the timeline. If set to None, the timeline will get
         its length from the keyframes.
    loop : bool, optional
        the number of legs the animal has (default 4)
    motion_path_res : int, default: None
        the number of line segments used to visualizer the timeline's motion
         path.
    """

    def __init__(self, actors=None, playback_panel=False, length=None,
                 loop=False, motion_path_res=None):

        super().__init__()
        self._data = defaultdict(dict)
        self._camera_data = defaultdict(dict)
        self.playback_panel = None
        self._last_timestamp = 0
        self._current_timestamp = 0
        self._speed = 1
        self._timelines = []
        self._static_actors = []
        self._camera = None
        self._scene = None
        self._last_started_time = 0
        self._playing = False
        self._length = length
        self._final_timestamp = 0
        self._needs_update = False
        self._reverse_playing = False
        self._loop = loop
        self._added_to_scene = True
        self._add_to_scene_time = 0
        self._remove_from_scene_time = None
        self._is_camera_animated = False
        self._motion_path_res = motion_path_res
        self._motion_path_actor = None
        self._parent_timeline = None

        # Handle actors while constructing the timeline.
        if playback_panel:
            def set_loop(loop):
                self._loop = loop

            def set_speed(speed):
                self.speed = speed

            self.playback_panel = PlaybackPanel(loop=self._loop)
            self.playback_panel.on_play = self.play
            self.playback_panel.on_stop = self.stop
            self.playback_panel.on_pause = self.pause
            self.playback_panel.on_loop_toggle = set_loop
            self.playback_panel.on_progress_bar_changed = self.seek
            self.playback_panel.on_speed_changed = set_speed
            self.add_actor(self.playback_panel, static=True)

        if actors is not None:
            self.add_actor(actors)

    def update_final_timestamp(self):
        """Calculates and returns the final timestamp of all keyframes.

        Returns
        -------
        float
            final timestamp that can be reached inside the Timeline.
        """
        if self._length is None:
            self._final_timestamp = max(self.final_timestamp,
                                        max([0] + [tl.update_final_timestamp()
                                                   for tl in self.timelines]))
        else:
            self._final_timestamp = self._length
        if self.has_playback_panel:
            self.playback_panel.final_time = self._final_timestamp
        return self._final_timestamp

    def update_motion_path(self):
        res = self._motion_path_res
        tl = self
        while not res and isinstance(tl._parent_timeline, Timeline):
            tl = tl._parent_timeline
            res = tl._motion_path_res
        if not res:
            return
        lines = []
        colors = []
        if self.is_interpolatable('position'):
            ts = np.linspace(0, self.final_timestamp, res)
            [lines.append(self.get_position(t).tolist()) for t in ts]
            if self.is_interpolatable('color'):
                [colors.append(self.get_color(t)) for t in ts]
            elif len(self.items) == 1:
                colors = sum([i.vcolors[0] / 255 for i in self.items]) / \
                         len(self.items)
            else:
                colors = [1, 1, 1]
        if len(lines) > 0:
            lines = np.array([lines])
            if colors is []:
                colors = np.array([colors])

            mpa = actor.line(lines, colors=colors, opacity=0.6)
            if self._scene:
                # remove old motion path actor
                if self._motion_path_actor is not None:
                    self._scene.rm(self._motion_path_actor)
                self._scene.add(mpa)
            self._motion_path_actor = mpa

    def set_timestamp(self, timestamp):
        """Set the current timestamp of the animation.

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

    def _get_data(self, is_camera=False):
        if is_camera:
            self._is_camera_animated = True
            return self._camera_data
        else:
            return self._data

    def _get_attribute_data(self, attrib, is_camera=False):
        data = self._get_data(is_camera=is_camera)

        if attrib not in data:
            data[attrib] = {
                'keyframes': defaultdict(dict),
                'interpolator': {
                    'base': linear_interpolator,
                    'func': None,
                    'args': defaultdict()
                },
            }
        return data.get(attrib)

    def set_keyframe(self, attrib, timestamp, value, is_camera=False,
                     update_interpolator=True, **kwargs):
        """Set a keyframe for a certain attribute.

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: ndarray
            Value of the keyframe at the given timestamp.
        is_camera: bool, optional
            Indicated whether setting a camera property or general property.
        update_interpolator: bool, optional
            Interpolator will be reinitialized if Ture

        Other Parameters
        ----------------
        pre_cp: ndarray, shape (1, M), optional
            The control point in case of using `cubic Bézier interpolator` when
            time exceeds this timestamp.
        post_cp: ndarray, shape (1, M), optional
            The control point in case of using `cubic Bézier interpolator` when
            time precedes this timestamp.
        in_tangent: ndarray, shape (1, M), optional
            The in tangent at that position for the cubic spline curve.
        out_tangent: ndarray, shape (1, M), optional
            The out tangent at that position for the cubic spline curve.
        """

        attrib_data = self._get_attribute_data(attrib, is_camera=is_camera)
        keyframes = attrib_data.get('keyframes')

        keyframes[timestamp] = {
            'value': np.array(value).astype(float),
            **{par: np.array(val).astype(float) for par, val in kwargs.items()
               if val is not None}
        }

        if update_interpolator:
            interp = attrib_data.get('interpolator')
            interp_base = interp.get('base')
            if interp_base is not None:
                interp_args = interp.get('args')
                new_interp = interp_base(keyframes, **interp_args)
                interp['func'] = new_interp

        if timestamp > self.final_timestamp:
            self._final_timestamp = timestamp
            if self.has_playback_panel:
                final_t = self.update_final_timestamp()
                self.playback_panel.final_time = final_t

        if timestamp > 0:
            self.update_animation(force=True)

        # update motion path
        self.update_motion_path()

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
        Keyframes can be on any of the following forms:
        >>> key_frames_simple = {1: [1, 2, 1], 2: [3, 4, 5]}
        >>> key_frames_bezier = {1: {'value': [1, 2, 1]},
        >>>                       2: {'value': [3, 4, 5], 'pre_cp': [1, 2, 3]}}

        Examples
        ---------
        >>> pos_keyframes = {1: np.array([1, 2, 3]), 3: np.array([5, 5, 5])}
        >>> Timeline.set_keyframes('position', pos_keyframes)
        """
        for t, data in keyframes.items():
            keyframe = keyframes.get(t)
            if isinstance(keyframe, dict):
                self.set_keyframe(attrib, t, **keyframe, is_camera=is_camera)
            else:
                self.set_keyframe(attrib, t, keyframe, is_camera=is_camera)

    def set_camera_keyframe(self, attrib, timestamp, value, **kwargs):
        """Set a keyframe for a camera property

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: float
            Value of the keyframe at the given timestamp.
        **kwargs: dict, optional
            Additional keyword arguments passed to `set_keyframe`.
        """
        self.set_keyframe(attrib, timestamp, value, is_camera=True, **kwargs)

    def is_inside_scene_at(self, timestamp):
        if self._remove_from_scene_time is not None and \
                timestamp >= self._remove_from_scene_time:
            return False
        elif timestamp >= self._add_to_scene_time:
            return True
        return False

    def add_to_scene_at(self, timestamp):
        """Set timestamp for adding Timeline to scene event.

        Parameters
        ----------
        timestamp: float
            Timestamp of the event.
        """
        self._add_to_scene_time = timestamp

    def remove_from_scene_at(self, timestamp):
        """Set timestamp for removing Timeline to scene event.

        Parameters
        ----------
        timestamp: float
            Timestamp of the event.
        """
        self._remove_from_scene_time = timestamp

    def handle_scene_event(self, in_scene):
        if self._scene is not None:
            if in_scene and not self._added_to_scene:
                super(Timeline, self).add_to_scene(self._scene)
                self._added_to_scene = True
            elif not in_scene and self._added_to_scene:
                super(Timeline, self).remove_from_scene(self._scene)
                self._added_to_scene = False

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
        Cubic Bézier curve control points are not supported yet in this setter.

        Examples
        ---------
        >>> cam_pos = {1: np.array([1, 2, 3]), 3: np.array([5, 5, 5])}
        >>> Timeline.set_camera_keyframes('position', cam_pos)
        """
        self.set_keyframes(attrib, keyframes, is_camera=True)

    def set_interpolator(self, attrib, interpolator, is_camera=False,
                         time_only=False, **kwargs):
        """Set keyframes interpolator for a certain property

        Parameters
        ----------
        attrib: str
            The name of the property.
        interpolator: function
            The generator function of the interpolator to be used to
            interpolate/evaluate keyframes.
        is_camera: bool, optional
            Indicated whether dealing with a camera property or general
            property.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function such as:
            >>> def get_position(t):
            >>>     return np.array([np.sin(t), np.cos(t) * 5, 5])

        Other Parameters
        ----------------
        spline_degree: int, optional
            The degree of the spline in case of setting a spline interpolator.

        Notes
        -----
        If this evaluator is used to evaluate the value of actor's
        properties such as position, scale, color, rotation, or opacity, it has
        to return a value thathas the same shape as the evaluated property,
        i.e.: for scale, it has to return an array with shape 1x3, and for
        opacity, it has to return a 1x1, an int, or a float value.

        Examples
        ---------
        >>> Timeline.set_interpolator('position', linear_interpolator)

        >>> pos_fun = lambda t: np.array([np.sin(t), np.cos(t), 0])
        >>> Timeline.set_interpolator('position', pos_fun)
        """

        attrib_data = self._get_attribute_data(attrib, is_camera=is_camera)
        keyframes = attrib_data.get('keyframes')
        interp_data = attrib_data.get('interpolator')
        if time_only:
            interp_data['base'] = None
            interp_data['func'] = interpolator
        else:
            interp_data['base'] = interpolator
            interp_data['args'] = kwargs
            new_interp = interpolator(keyframes, **kwargs)
            interp_data['func'] = new_interp
        # update motion path
        self.update_motion_path()

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
        True means that it's safe to use `Interpolator.interpolate(t)` for the
        specified property. And False means the opposite.

        """
        data = self._camera_data if is_camera else self._data
        return attrib in data

    def set_camera_interpolator(self, attrib, interpolator, time_only=False):
        """Set the interpolator for a specific camera property.

        Parameters
        ----------
        attrib: str
            The name of the camera property.
            The already handled properties are position, focal, and view_up.

        interpolator: function
            The generator function of the interpolator that handles the
            camera property interpolation between keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.

        Examples
        ---------
        >>> Timeline.set_camera_interpolator('focal', linear_interpolator)
        """
        self.set_interpolator(attrib, interpolator, is_camera=True,
                              time_only=time_only)

    def set_position_interpolator(self, interpolator, time_only=False,
                                  **kwargs):
        """Set the position interpolator for all actors inside the
        timeline.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
             position keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.

        Other Parameters
        ----------------
        degree: int
            The degree of the spline interpolation in case of setting
            the `spline_interpolator`.

        Examples
        ---------
        >>> Timeline.set_position_interpolator(spline_interpolator, degree=5)
        """
        self.set_interpolator('position', interpolator,
                              time_only=time_only, **kwargs)

    def set_scale_interpolator(self, interpolator, time_only=False):
        """Set the scale interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: function
            TThe generator function of the interpolator that would handle
            the scale keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.

        Examples
        ---------
        >>> Timeline.set_scale_interpolator(step_interpolator)
        """
        self.set_interpolator('scale', interpolator, time_only=time_only)

    def set_rotation_interpolator(self, interpolator, time_only=False):
        """Set the scale interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            rotation (orientation) keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.
        Examples
        ---------
        >>> Timeline.set_rotation_interpolator(slerp)
        """
        self.set_interpolator('rotation', interpolator, time_only=time_only)

    def set_color_interpolator(self, interpolator, time_only=False):
        """Set the color interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle
            the color keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.
        Examples
        ---------
        >>> Timeline.set_color_interpolator(lab_color_interpolator)
        """
        self.set_interpolator('color', interpolator, time_only=time_only)

    def set_opacity_interpolator(self, interpolator, time_only=False):
        """Set the opacity interpolator for all the actors inside the
        timeline.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle
            the opacity keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.
        Examples
        ---------
        >>> Timeline.set_opacity_interpolator(step_interpolator)
        """
        self.set_interpolator('opacity', interpolator, time_only=time_only)

    def set_camera_position_interpolator(self, interpolator, time_only=False):
        """Set the camera position interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            interpolation of the camera position keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.
        """
        self.set_camera_interpolator("position", interpolator,
                                     time_only=time_only)

    def set_camera_focal_interpolator(self, interpolator, time_only=False):
        """Set the camera focal position interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            interpolation of the camera focal position keyframes.
        time_only: bool, optional
            `True` if `interpolator` is time-only based function.
        """
        self.set_camera_interpolator("focal", interpolator,
                                     time_only=time_only)

    def get_value(self, attrib, timestamp):
        """Returns the value of an attribute at any given timestamp.

        Parameters
        ----------
        attrib: str
            The attribute name.
        timestamp: float
            The timestamp to interpolate at.
        """
        return self._data.get(attrib).get('interpolator').\
            get('func')(timestamp)

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
        return self._camera_data.get(attrib).get('interpolator').\
            get('func')(timestamp)

    def set_position(self, timestamp, position, **kwargs):
        """Set a position keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        position: ndarray, shape (1, 3)
            Position value

        Other Parameters
        ----------------
        pre_cp: float
            The control point in case of using `cubic Bézier interpolator` when
            time exceeds this timestamp.
        post_cp: float
            The control point in case of using `cubic Bézier interpolator` when
            time precedes this timestamp.
        in_tangent: ndarray, shape (1, M), optional
            The in tangent at that position for the cubic spline curve.
        out_tangent: ndarray, shape (1, M), optional
            The out tangent at that position for the cubic spline curve.

        Notes
        -----
        `pre_cp` and `post_cp` only needed when using the cubic bezier
        interpolation method.
        """
        self.set_keyframe('position', timestamp, position, **kwargs)

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

    def set_rotation(self, timestamp, rotation, **kwargs):
        """Set a rotation keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        rotation: ndarray, shape(1, 3) or shape(1, 4)
            Rotation data in euler degrees with shape(1, 3) or in quaternions
            with shape(1, 4).

        Notes
        -----
        Euler rotations are executed by rotating first around Z then around X,
        and finally around Y.
        """
        no_components = len(np.array(rotation).flatten())
        if no_components == 4:
            self.set_keyframe('rotation', timestamp, rotation, **kwargs)
        elif no_components == 3:
            # user is expected to set rotation order by default as setting
            # orientation of a `vtkActor` ordered as z->x->y.
            rotation = np.asarray(rotation, dtype=float)
            rotation = transform.Rotation.from_euler('zxy',
                                                     rotation[[2, 0, 1]],
                                                     degrees=True).as_quat()
            self.set_keyframe('rotation', timestamp, rotation, **kwargs)
        else:
            warnings.warn(f'Keyframe with {no_components} components is not a '
                          f'valid rotation data. Skipped!')

    def set_rotation_as_vector(self, timestamp, vector, **kwargs):
        """Set a rotation keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        vector: ndarray, shape(1, 3)
            Directional vector that describes the rotation.
        """
        quat = transform.Rotation.from_rotvec(vector).as_quat()
        self.set_keyframe('rotation', timestamp, quat, **kwargs)

    def set_scale(self, timestamp, scalar, **kwargs):
        """Set a scale keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        scalar: ndarray, shape(1, 3)
            Scale keyframe value associated with the timestamp.
        """
        self.set_keyframe('scale', timestamp, scalar, **kwargs)

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

    def set_color(self, timestamp, color, **kwargs):
        """Set color keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        color: ndarray, shape(1, 3)
            Color keyframe value associated with the timestamp.
        """
        self.set_keyframe('color', timestamp, color, **kwargs)

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

    def set_opacity(self, timestamp, opacity, **kwargs):
        """Set opacity keyframe at a specific timestamp.

        Parameters
        ----------
        timestamp: float
            Timestamp of the keyframe
        opacity: ndarray, shape(1, 3)
            Opacity keyframe value associated with the timestamp.
        """
        self.set_keyframe('opacity', timestamp, opacity, **kwargs)

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

    def get_rotation(self, t, as_quat=False):
        """Returns the interpolated rotation.

        Parameters
        ----------
        t: float
            the time to interpolate rotation at.
        as_quat: bool
            Returned rotation will be as quaternion if True.

        Returns
        -------
        ndarray(1, 3):
            The interpolated rotation as Euler degrees by default.
        """
        rot = self.get_value('rotation', t)
        if len(rot) == 4:
            if as_quat:
                return rot
            r = transform.Rotation.from_quat(rot)
            degrees = r.as_euler('zxy', degrees=True)[[1, 2, 0]]
            return degrees
        elif not as_quat:
            return rot
        return transform.Rotation.from_euler('zxy',
                                             rot[[2, 0, 1]],
                                             degrees=True).as_quat()

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

    def set_camera_position(self, timestamp, position, **kwargs):
        """Sets the camera position keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate opacity at.
        position: ndarray, shape(1, 3)
            The camera position
        """
        self.set_camera_keyframe('position', timestamp, position, **kwargs)

    def set_camera_focal(self, timestamp, position, **kwargs):
        """Sets camera's focal position keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate opacity at.
        position: ndarray, shape(1, 3)
            The camera position
        """
        self.set_camera_keyframe('focal', timestamp, position, **kwargs)

    def set_camera_view_up(self, timestamp, direction, **kwargs):
        """Sets the camera view-up direction keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate at.
        direction: ndarray, shape(1, 3)
            The camera view-up direction
        """
        self.set_camera_keyframe('view_up', timestamp, direction, **kwargs)

    def set_camera_rotation(self, timestamp, rotation, **kwargs):
        """Sets the camera rotation keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate at.
        rotation: ndarray, shape(1, 3) or shape(1, 4)
            Rotation data in euler degrees with shape(1, 3) or in quaternions
            with shape(1, 4).

        Notes
        -----
        Euler rotations are executed by rotating first around Z then around X,
        and finally around Y.
        """
        self.set_rotation(timestamp, rotation, is_camera=True, **kwargs)

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
        >>> Timeline.set_camera_view_up_keyframes(view_ups)
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
        elif isinstance(item, Actor):
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
        timeline._parent_timeline = self
        timeline.update_motion_path()
        self._timelines.append(timeline)

    def add_actor(self, actor, static=False):
        """Adds an actor or list of actors to the Timeline.

        Parameters
        ----------
        actor: vtkActor or list(vtkActor)
            Actor/s to be animated by the timeline.
        static: bool
            Indicated whether the actor should be animated and controlled by
            the timeline or just a static actor that gets added to the scene
            along with the Timeline.
        """
        if isinstance(actor, list):
            for a in actor:
                self.add_actor(a, static=static)
        elif static:
            self._static_actors.append(actor)
        else:
            actor.vcolors = utils.colors_from_actor(actor)
            super(Timeline, self).add(actor)

    @property
    def parent_timeline(self):
        return self._parent_timeline

    @property
    def actors(self):
        """Returns a list of actors.

        Returns
        -------
        list:
            List of actors controlled by the Timeline.
        """
        return self.items

    @property
    def timelines(self):
        """Returns a list of child Timelines.

        Returns
        -------
        list:
            List of child Timelines of this Timeline.
        """
        return self._timelines

    def add_static_actor(self, actor):
        """Adds an actor or list of actors as static actor/s which will not be
        controlled nor animated by the Timeline. All static actors will be
        added to the scene when the Timeline is added to the scene.

        Parameters
        ----------
        actor: vtkActor or list(vtkActor)
            Static actor/s.
        """
        self.add_actor(actor, static=True)

    @property
    def static_actors(self):
        """Returns a list of static actors.

        Returns
        -------
        list:
            List of static actors.
        """
        return self._static_actors

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

    def update_animation(self, t=None, force=False, _in_scene=None):
        """Updates the timeline animations"""
        if t is None:
            t = self.current_timestamp
            if t > self._final_timestamp:
                if self._loop:
                    self.seek(0)
                else:
                    self.seek(self.final_timestamp)
                    # Doing this will pause both the timeline and the panel.
                    self.playback_panel.pause()
        if self.has_playback_panel and (self.playing or force):
            self.update_final_timestamp()
            self.playback_panel.current_time = t

        # handling in/out of scene events
        in_scene = self.is_inside_scene_at(t)
        if _in_scene is not None:
            in_scene = _in_scene and in_scene
        self.handle_scene_event(in_scene)

        if self.playing or force:
            if self._camera is not None:
                if self.is_interpolatable('rotation', is_camera=True):
                    pos = self._camera.GetPosition()
                    translation = np.identity(4)
                    translation[:3, 3] = pos
                    # camera axis is reverted
                    rot = -self.get_camera_rotation(t)
                    rot = transform.Rotation.from_quat(rot).as_matrix()
                    rot = np.array([[*rot[0], 0],
                                    [*rot[1], 0],
                                    [*rot[2], 0],
                                    [0, 0, 0, 1]])
                    rot = translation @ rot @ np.linalg.inv(translation)
                    self._camera.SetModelTransformMatrix(rot.flatten())

                if self.is_interpolatable('position', is_camera=True):
                    cam_pos = self.get_camera_position(t)
                    self._camera.SetPosition(cam_pos)

                if self.is_interpolatable('focal', is_camera=True):
                    cam_foc = self.get_camera_focal(t)
                    self._camera.SetFocalPoint(cam_foc)

                if self.is_interpolatable('view_up', is_camera=True):
                    cam_up = self.get_camera_view_up(t)
                    self._camera.SetViewUp(cam_up)
                elif not self.is_interpolatable('view_up', is_camera=True):
                    # to preserve up-view as default after user interaction
                    self._camera.SetViewUp(0, 1, 0)

            elif self._is_camera_animated and self._scene:
                self._camera = self._scene.camera()
                self.update_animation(force=True)
                return

            # actors properties
            if in_scene:
                if self.is_interpolatable('position'):
                    position = self.get_position(t)
                    self.SetPosition(position)

                if self.is_interpolatable('scale'):
                    scale = self.get_scale(t)
                    [act.SetScale(scale) for act in self.actors]

                if self.is_interpolatable('opacity'):
                    scale = self.get_opacity(t)
                    [act.GetProperty().SetOpacity(scale) for
                     act in self.actors]

                if self.is_interpolatable('rotation'):
                    euler = self.get_rotation(t)
                    [act.SetOrientation(euler) for
                     act in self.actors]

                if self.is_interpolatable('color'):
                    color = self.get_color(t)
                    for act in self.actors:
                        act.vcolors[:] = color * 255
                        utils.update_actor(act)
                # Also update all child Timelines.
            [tl.update_animation(t, force=True, _in_scene=in_scene)
             for tl in self.timelines]
            # update clipping range
            if self.parent_timeline is None and self._scene:
                self._scene.reset_clipping_range()

    def play(self):
        """Play the animation"""
        if not self.playing:
            if self.current_timestamp >= self.final_timestamp:
                self.current_timestamp = 0
            self.update_final_timestamp()
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
        """Set the current timestamp of the Timeline.

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
        # assuring timestamp value is in the timeline range
        if timestamp < 0:
            timestamp = 0
        elif timestamp > self.final_timestamp:
            timestamp = self.final_timestamp

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
        """Returns whether the Timeline is playing.

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
        """Returns whether the Timeline is stopped.

        Returns
        -------
        bool
            Timeline is stopped if True.

        """
        return not self.playing and not self._last_timestamp

    @property
    def paused(self):
        """Returns whether the Timeline is paused.

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
        current = self.current_timestamp
        if speed <= 0:
            return
        self._speed = speed
        self._last_started_time = time.perf_counter()
        self.current_timestamp = current

    @property
    def has_playback_panel(self):
        return self.playback_panel is not None

    def add_to_scene(self, ren):
        super(Timeline, self).add_to_scene(ren)
        [ren.add(static_act) for static_act in self._static_actors]
        [ren.add(timeline) for timeline in self.timelines]
        if self._motion_path_actor:
            ren.add(self._motion_path_actor)
        self._scene = ren
        self._added_to_scene = True
        self.update_animation(force=True)
