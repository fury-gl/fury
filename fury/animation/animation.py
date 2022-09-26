import time
import warnings
from collections import defaultdict
from fury import utils, actor
from fury.actor import Container
from fury.animation.interpolator import spline_interpolator, \
    step_interpolator, linear_interpolator, slerp
import numpy as np
from scipy.spatial import transform
from fury.ui.elements import PlaybackPanel
from fury.lib import Actor, Transform


class Animation(Container):
    """Keyframe animation class.

    Animation is responsible for keyframe animations for a single or a
    group of actors.
    It's used to handle multiple attributes and properties of Fury actors such
    as transformations, color, and scale.
    It also accepts custom data and interpolates them, such as temperature.
    Linear interpolation is used by default to interpolate data between the
    main keyframes.

    Attributes
    ----------
    actors : Actor or list[Actor], optional, default: None
        Actor/s to be animated.
    length : float or int, default: None, optional
        the fixed length of the animation. If set to None, the animation will
        get its duration from the keyframes being set.
    loop : bool, optional
        Whether to loop the animation (True) of play once (False).
    motion_path_res : int, default: None
        the number of line segments used to visualizer the animation's motion
        path (visualizing position).
    """

    def __init__(self, actors=None, length=None, loop=False,
                 motion_path_res=None):

        super().__init__()
        self._data = defaultdict(dict)
        self._camera_data = defaultdict(dict)
        self._animations = []
        self._static_actors = []
        self._timeline = None
        self._parent_animation = None
        self._camera = None
        self._scene = None
        self._added_to_scene_time = 0
        self._length = length
        self._duration = length if length else 0
        self._loop = loop
        self._max_timestamp = 0
        self._added_to_scene = True
        self._is_camera_animated = False
        self._motion_path_res = motion_path_res
        self._motion_path_actor = None
        self._transform = Transform()

        # Adding actors to the animation
        if actors is not None:
            self.add_actor(actors)

    def update_duration(self):
        """Update and return the duration of the Animation.

        Returns
        -------
        float
            The duration of the animation.
        """
        if self._length is not None:
            self._duration = self._length
        else:
            self._duration = max(
                self._max_timestamp, max([0] + [anim.update_duration() for anim
                                                in self.child_animations]))
        return self.duration

    @property
    def duration(self):
        """Return the duration of the animation.

        Returns
        -------
        float
            The duration of the animation.
        """
        return self._duration

    def update_motion_path(self):
        """Update motion path visualization actor"""
        res = self._motion_path_res
        tl = self
        while isinstance(tl._parent_animation, Animation):
            if res:
                break
            tl = tl._parent_animation
            res = tl._motion_path_res
        if not res:
            return

        lines = []
        colors = []
        if self.is_interpolatable('position'):
            ts = np.linspace(0, self._max_timestamp, res)
            [lines.append(self.get_position(t).tolist()) for t in ts]
            if self.is_interpolatable('color'):
                [colors.append(self.get_color(t)) for t in ts]
            elif len(self.items) >= 1:
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
                    'base': linear_interpolator if attrib != 'rotation' else
                    slerp,
                    'func': None,
                    'args': defaultdict()
                },
                'callbacks': [],
            }
        return data.get(attrib)

    def get_keyframes(self, attrib=None, is_camera=False):
        """Set a keyframe for a certain attribute.

        Parameters
        ----------
        attrib: str, optional, default: None
            The name of the attribute.
            If None, all keyframes for all set attributes will be returned.
        is_camera: bool, optional
            Indicated whether setting a camera property or general property.
        """
        
        data = self._get_data(is_camera=is_camera)
        if attrib is None:
            attribs = data.keys()
            return {attrib: data.get(attrib, {}).get('keyframes', {}) for 
                    attrib in attribs}
        return data.get(attrib, {}).get('keyframes', {})

    def set_keyframe(self, attrib, timestamp, value, is_camera=False,
                     update_interpolator=True, **kwargs):
        """Set a keyframe for a certain attribute.

        Parameters
        ----------
        attrib: str
            The name of the attribute.
        timestamp: float
            Timestamp of the keyframe.
        value: ndarray or float or bool
            Value of the keyframe at the given timestamp.
        is_camera: bool, optional
            Indicated whether setting a camera property or general property.
        update_interpolator: bool, optional
            Interpolator will be reinitialized if Ture

        Other Parameters
        ----------------
        in_cp: ndarray, shape (1, M), optional
            The in control point in case of using cubic Bézier interpolator.
        out_cp: ndarray, shape (1, M), optional
            The out control point in case of using cubic Bézier interpolator.
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
            interp_base = interp.get(
                'base', linear_interpolator if attrib != 'rotation' else slerp)
            args = interp.get('args', {})
            self.set_interpolator(attrib, interp_base,
                                  is_camera=is_camera, **args)

        if timestamp > self._max_timestamp:
            self._max_timestamp = timestamp
            if self._timeline is not None:
                self._timeline.update_duration()
            else:
                self.update_duration()
        self.update_animation()
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
        >>>                       2: {'value': [3, 4, 5], 'in_cp': [1, 2, 3]}}

        Examples
        ---------
        >>> pos_keyframes = {1: np.array([1, 2, 3]), 3: np.array([5, 5, 5])}
        >>> Animation.set_keyframes('position', pos_keyframes)
        """
        for t, keyframe in keyframes.items():
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
        value: value: ndarray or float or bool
            Value of the keyframe at the given timestamp.
        **kwargs: dict, optional
            Additional keyword arguments passed to `set_keyframe`.
        """
        self.set_keyframe(attrib, timestamp, value, is_camera=True, **kwargs)

    def is_inside_scene_at(self, timestamp):
        parent = self._parent_animation
        parent_in_scene = True
        if parent is not None:
            parent_in_scene = parent._added_to_scene

        if self.is_interpolatable('in_scene'):
            return parent_in_scene and self.get_value('in_scene', timestamp)

        return parent_in_scene

    def add_to_scene_at(self, timestamp):
        """Set timestamp for adding Animation to scene event.

        Parameters
        ----------
        timestamp: float
            Timestamp of the event.
        """
        if not self.is_interpolatable('in_scene'):
            self.set_keyframe('in_scene', timestamp, True)
            self.set_interpolator('in_scene', step_interpolator)
        else:
            self.set_keyframe('in_scene', timestamp, True)

    def remove_from_scene_at(self, timestamp):
        """Set timestamp for removing Animation to scene event.

        Parameters
        ----------
        timestamp: float
            Timestamp of the event.
        """
        if not self.is_interpolatable('in_scene'):
            self.set_keyframe('in_scene', timestamp, False)
            self.set_interpolator('in_scene', step_interpolator)
        else:
            self.set_keyframe('in_scene', timestamp, False)

    def _handle_scene_event(self, timestamp):
        should_be_in_scene = self.is_inside_scene_at(timestamp)
        if self._scene is not None:
            if should_be_in_scene and not self._added_to_scene:
                super(Animation, self).add_to_scene(self._scene)
                self._added_to_scene = True
            elif not should_be_in_scene and self._added_to_scene:
                super(Animation, self).remove_from_scene(self._scene)
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
        >>> Animation.set_camera_keyframes('position', cam_pos)
        """
        self.set_keyframes(attrib, keyframes, is_camera=True)

    def set_interpolator(self, attrib, interpolator, is_camera=False,
                         is_evaluator=False, **kwargs):
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
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes such as:
            >>> def get_position(t):
            >>>     return np.array([np.sin(t), np.cos(t) * 5, 5])

        Other Parameters
        ----------------
        spline_degree: int, optional
            The degree of the spline in case of setting a spline interpolator.

        Notes
        -----
        If an evaluator is used to set the values of actor's properties such as
        position, scale, color, rotation, or opacity, it has to return a value
        with the same shape as the evaluated property, i.e.: for scale, it
        has to return an array with shape 1x3, and for opacity, it has to
        return a 1x1, an int, or a float value.

        Examples
        ---------
        >>> Animation.set_interpolator('position', linear_interpolator)

        >>> pos_fun = lambda t: np.array([np.sin(t), np.cos(t), 0])
        >>> Animation.set_interpolator('position', pos_fun)
        """

        attrib_data = self._get_attribute_data(attrib, is_camera=is_camera)
        keyframes = attrib_data.get('keyframes', {})
        interp_data = attrib_data.get('interpolator', {})
        if is_evaluator:
            interp_data['base'] = None
            interp_data['func'] = interpolator
        else:
            interp_data['base'] = interpolator
            interp_data['args'] = kwargs
            # Maintain interpolator base incase new keyframes are added.
            if len(keyframes) == 0:
                return
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
            True if the property is interpolatable by the Animation.

        Notes
        -------
        True means that it's safe to use `Interpolator.interpolate(t)` for the
        specified property. And False means the opposite.

        """
        data = self._camera_data if is_camera else self._data
        return bool(data.get(attrib, {}).get('interpolator', {}).get('func'))

    def set_camera_interpolator(self, attrib, interpolator,
                                is_evaluator=False):
        """Set the interpolator for a specific camera property.

        Parameters
        ----------
        attrib: str
            The name of the camera property.
            The already handled properties are position, focal, and view_up.

        interpolator: function
            The generator function of the interpolator that handles the
            camera property interpolation between keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.

        Examples
        ---------
        >>> Animation.set_camera_interpolator('focal', linear_interpolator)
        """
        self.set_interpolator(attrib, interpolator, is_camera=True,
                              is_evaluator=is_evaluator)

    def set_position_interpolator(self, interpolator, is_evaluator=False,
                                  **kwargs):
        """Set the position interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
             position keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.

        Other Parameters
        ----------------
        degree: int
            The degree of the spline interpolation in case of setting
            the `spline_interpolator`.

        Examples
        ---------
        >>> Animation.set_position_interpolator(spline_interpolator, degree=5)
        """
        self.set_interpolator('position', interpolator,
                              is_evaluator=is_evaluator, **kwargs)

    def set_scale_interpolator(self, interpolator, is_evaluator=False):
        """Set the scale interpolator.

        Parameters
        ----------
        interpolator: function
            TThe generator function of the interpolator that would handle
            the scale keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.

        Examples
        ---------
        >>> Animation.set_scale_interpolator(step_interpolator)
        """
        self.set_interpolator('scale', interpolator, is_evaluator=is_evaluator)

    def set_rotation_interpolator(self, interpolator, is_evaluator=False):
        """Set the rotation interpolator .

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            rotation (orientation) keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.
        Examples
        ---------
        >>> Animation.set_rotation_interpolator(slerp)
        """
        self.set_interpolator('rotation', interpolator,
                              is_evaluator=is_evaluator)

    def set_color_interpolator(self, interpolator, is_evaluator=False):
        """Set the color interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle
            the color keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.
        Examples
        ---------
        >>> Animation.set_color_interpolator(lab_color_interpolator)
        """
        self.set_interpolator('color', interpolator, is_evaluator=is_evaluator)

    def set_opacity_interpolator(self, interpolator, is_evaluator=False):
        """Set the opacity interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle
            the opacity keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.
        Examples
        ---------
        >>> Animation.set_opacity_interpolator(step_interpolator)
        """
        self.set_interpolator('opacity', interpolator,
                              is_evaluator=is_evaluator)

    def set_camera_position_interpolator(self, interpolator,
                                         is_evaluator=False):
        """Set the camera position interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            interpolation of the camera position keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.
        """
        self.set_camera_interpolator("position", interpolator,
                                     is_evaluator=is_evaluator)

    def set_camera_focal_interpolator(self, interpolator, is_evaluator=False):
        """Set the camera focal position interpolator.

        Parameters
        ----------
        interpolator: function
            The generator function of the interpolator that would handle the
            interpolation of the camera focal position keyframes.
        is_evaluator: bool, optional
            Specifies whether the `interpolator` is time-only based evaluation
            function that does not depend on keyframes.
        """
        self.set_camera_interpolator("focal", interpolator,
                                     is_evaluator=is_evaluator)

    def get_value(self, attrib, timestamp):
        """Return the value of an attribute at any given timestamp.

        Parameters
        ----------
        attrib: str
            The attribute name.
        timestamp: float
            The timestamp to interpolate at.
        """
        value = self._data.get(attrib, {}).get('interpolator', {}). \
            get('func')(timestamp)
        return value

    def get_current_value(self, attrib):
        """Return the value of an attribute at current time.

        Parameters
        ----------
        attrib: str
            The attribute name.
        """
        return self._data.get(attrib).get('interpolator'). \
            get('func')(self._timeline.current_timestamp)

    def get_camera_value(self, attrib, timestamp):
        """Return the value of an attribute interpolated at any given
        timestamp.

        Parameters
        ----------
        attrib: str
            The attribute name.
        timestamp: float
            The timestamp to interpolate at.

        """
        return self._camera_data.get(attrib).get('interpolator'). \
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
        in_cp: float
            The control point in case of using `cubic Bézier interpolator` when
            time exceeds this timestamp.
        out_cp: float
            The control point in case of using `cubic Bézier interpolator` when
            time precedes this timestamp.
        in_tangent: ndarray, shape (1, M), optional
            The in tangent at that position for the cubic spline curve.
        out_tangent: ndarray, shape (1, M), optional
            The out tangent at that position for the cubic spline curve.

        Notes
        -----
        `in_cp` and `out_cp` only needed when using the cubic bezier
        interpolation method.
        """
        self.set_keyframe('position', timestamp, position, **kwargs)

    def set_position_keyframes(self, keyframes):
        """Set a dict of position keyframes at once.
        Should be in the following form:
        {timestamp_1: position_1, timestamp_2: position_2}

        Parameters
        ----------
        keyframes: dict
            A dict with timestamps as keys and positions as values.

        Examples
        --------
        >>> pos_keyframes = {1, np.array([0, 0, 0]), 3, np.array([50, 6, 6])}
        >>> Animation.set_position_keyframes(pos_keyframes)
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
        keyframes: dict
            A dict with timestamps as keys and scales as values.

        Examples
        --------
        >>> scale_keyframes = {1, np.array([1, 1, 1]), 3, np.array([2, 2, 3])}
        >>> Animation.set_scale_keyframes(scale_keyframes)
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
        >>> Animation.set_color_keyframes(color_keyframes)
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
        >>> Animation.set_scale_keyframes(opacity)
        """
        self.set_keyframes('opacity', keyframes)

    def get_position(self, t):
        """Return the interpolated position.

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
        """Return the interpolated rotation.

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
        """Return the interpolated scale.

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
        """Return the interpolated color.

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
        """Return the opacity value.

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
        """Set the camera position keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate opacity at.
        position: ndarray, shape(1, 3)
            The camera position
        """
        self.set_camera_keyframe('position', timestamp, position, **kwargs)

    def set_camera_focal(self, timestamp, position, **kwargs):
        """Set camera's focal position keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate opacity at.
        position: ndarray, shape(1, 3)
            The camera position
        """
        self.set_camera_keyframe('focal', timestamp, position, **kwargs)

    def set_camera_view_up(self, timestamp, direction, **kwargs):
        """Set the camera view-up direction keyframe.

        Parameters
        ----------
        timestamp: float
            The time to interpolate at.
        direction: ndarray, shape(1, 3)
            The camera view-up direction
        """
        self.set_camera_keyframe('view_up', timestamp, direction, **kwargs)

    def set_camera_rotation(self, timestamp, rotation, **kwargs):
        """Set the camera rotation keyframe.

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
        keyframes: dict
            A dict with timestamps as keys and opacities as values.

        Examples
        --------
        >>> pos = {0, np.array([1, 1, 1]), 3, np.array([20, 0, 0])}
        >>> Animation.set_camera_position_keyframes(pos)
        """
        self.set_camera_keyframes('position', keyframes)

    def set_camera_focal_keyframes(self, keyframes):
        """Set multiple camera focal position keyframes at once.
        Should be in the following form:
        {timestamp_1: focal_1, timestamp_2: focal_1, ...}

        Parameters
        ----------
        keyframes: dict
            A dict with timestamps as keys and camera focal positions as
            values.

        Examples
        --------
        >>> focal_pos = {0, np.array([1, 1, 1]), 3, np.array([20, 0, 0])}
        >>> Animation.set_camera_focal_keyframes(focal_pos)
        """
        self.set_camera_keyframes('focal', keyframes)

    def set_camera_view_up_keyframes(self, keyframes):
        """Set multiple camera view up direction keyframes.
        Should be in the following form:
        {timestamp_1: view_up_1, timestamp_2: view_up_2, ...}

        Parameters
        ----------
        keyframes: dict
            A dict with timestamps as keys and camera view up vectors as
            values.

        Examples
        --------
        >>> view_ups = {0, np.array([1, 0, 0]), 3, np.array([0, 1, 0])}
        >>> Animation.set_camera_view_up_keyframes(view_ups)
        """
        self.set_camera_keyframes('view_up', keyframes)

    def get_camera_position(self, t):
        """Return the interpolated camera position.

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
        """Return the interpolated camera's focal position.

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
        """Return the interpolated camera's view-up directional vector.

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
        """Return the interpolated rotation for the camera expressed
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
        """Add an item to the Animation.
        This item can be an Actor, Animation, list of Actors, or a list of
        Animations.

        Parameters
        ----------
        item: Animation, vtkActor, list[Animation], or list[vtkActor]
            Actor/s to be animated by the Animation.
        """
        if isinstance(item, list):
            for a in item:
                self.add(a)
            return
        elif isinstance(item, Actor):
            self.add_actor(item)
        elif isinstance(item, Animation):
            self.add_child_animation(item)
        else:
            raise ValueError(f"Object of type {type(item)} can't be animated")

    def add_child_animation(self, animation):
        """Add child Animation or list of Animations.

        Parameters
        ----------
        animation: Animation or list[Animation]
            Animation/s to be added.
        """
        if isinstance(animation, list):
            for a in animation:
                self.add_child_animation(a)
            return
        animation._parent_animation = self
        animation.update_motion_path()
        self._animations.append(animation)
        self.update_duration()

    def add_actor(self, actor, static=False):
        """Add an actor or list of actors to the Animation.

        Parameters
        ----------
        actor: vtkActor or list(vtkActor)
            Actor/s to be animated by the Animation.
        static: bool
            Indicated whether the actor should be animated and controlled by
            the animation or just a static actor that gets added to the scene
            along with the Animation.
        """
        if isinstance(actor, list):
            for a in actor:
                self.add_actor(a, static=static)
        elif static:
            self._static_actors.append(actor)
        else:
            actor.vcolors = utils.colors_from_actor(actor)
            super(Animation, self).add(actor)

    @property
    def timeline(self):
        """Return the Timeline handling the current animation.

        Returns
        -------
        Timeline:
            The Timeline handling the current animation, None, if there is no
            associated Timeline.

        """
        return self._timeline

    @timeline.setter
    def timeline(self, timeline):
        """Assign the Timeline responsible for handling the Animation.

        Parameters
        ----------

        Timeline:
            The Timeline handling the current animation, None, if there is no
            associated Timeline.

        """
        self._timeline = timeline
        if self._animations:
            for animation in self._animations:
                animation.timeline = timeline

    @property
    def parent_animation(self):
        """Return the hierarchical parent Animation for current Animation.

        Returns
        -------
        Animation:
            The parent Animation.

        """
        return self._parent_animation

    @parent_animation.setter
    def parent_animation(self, parent_animation):
        """Assign a parent Animation for the current Animation.

        Parameters
        ----------
        parent_animation: Animation
            The parent Animation instance.
        """
        self._parent_animation = parent_animation

    @property
    def actors(self):
        """Return a list of actors.

        Returns
        -------
        list:
            List of actors controlled by the Animation.
        """
        return self.items

    @property
    def child_animations(self) -> 'list[Animation]':
        """Return a list of child Animations.

        Returns
        -------
        list:
            List of child Animations of this Animation.
        """
        return self._animations

    def add_static_actor(self, actor):
        """Add an actor or list of actors as static actor/s which will not be
        controlled nor animated by the Animation. All static actors will be
        added to the scene when the Animation is added to the scene.

        Parameters
        ----------
        actor: vtkActor or list(vtkActor)
            Static actor/s.
        """
        self.add_actor(actor, static=True)

    @property
    def static_actors(self):
        """Return a list of static actors.

        Returns
        -------
        list:
            List of static actors.
        """
        return self._static_actors

    def remove_animations(self):
        """Remove all child Animations from the Animation"""
        self._animations.clear()

    def remove_actor(self, actor):
        """Remove an actor from the Animation.

        Parameters
        ----------
        actor: vtkActor
            Actor to be removed from the Animation.
        """
        self._items.remove(actor)

    def remove_actors(self):
        """Remove all actors from the Animation"""
        self.clear()

    def add_update_callback(self, property_name, cbk_func):
        """Add a function to be called each time animation is updated
        This function must accept only one argument which is the current value
        of the named property.


        Parameters
        ----------
        property_name: str
            The name of the property.
        cbk_func: function
            The function to be called whenever the animation is updated.
        """
        attrib = self._get_attribute_data(property_name)
        attrib.get('callbacks', []).append(cbk_func)

    def update_animation(self, t=None):
        """Update the animation.

        Update the animation at a certain time. This will make sure all
        attributes are calculated and set to the actors at that given time.

        Parameters
        ----------
        t: float or int, optional, default: None
            Time to update animation at.
        """
        t = t if t is not None else \
            time.perf_counter() - self._added_to_scene_time

        # handling in/out of scene events
        in_scene = self.is_inside_scene_at(t)
        self._handle_scene_event(t)

        if self.duration:
            if self._loop and t > 0:
                t = t % self.duration
            elif t > self.duration:
                t = self.duration
        if isinstance(self._parent_animation, Animation):
            self._transform.DeepCopy(self._parent_animation._transform)
        else:
            self._transform.Identity()

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
            self.update_animation(t)
            return

        # actors properties
        if in_scene:
            if self.is_interpolatable('position'):
                position = self.get_position(t)
                self._transform.Translate(*position)

            if self.is_interpolatable('opacity'):
                opacity = self.get_opacity(t)
                [act.GetProperty().SetOpacity(opacity) for
                 act in self.actors]

            if self.is_interpolatable('rotation'):
                x, y, z = self.get_rotation(t)
                # Rotate in the same order as VTK defaults.
                self._transform.RotateZ(z)
                self._transform.RotateX(x)
                self._transform.RotateY(y)

            if self.is_interpolatable('scale'):
                scale = self.get_scale(t)
                self._transform.Scale(*scale)

            if self.is_interpolatable('color'):
                color = self.get_color(t)
                for act in self.actors:
                    act.vcolors[:] = color * 255
                    utils.update_actor(act)

            # update actors' transformation matrix
            [act.SetUserTransform(self._transform) for act in self.actors]

        for attrib in self._data:
            callbacks = self._data.get(attrib, {}).get('callbacks', [])
            if callbacks is not [] and self.is_interpolatable(attrib):
                value = self.get_value(attrib, t)
                [cbk(value) for cbk in callbacks]

        # Also update all child Animations.
        [animation.update_animation(t) for animation in self._animations]

        if self._scene and self._parent_animation is None:
            self._scene.reset_clipping_range()

    def add_to_scene(self, ren):
        """Add this Animation, its actors and sub Animations to the scene"""
        super(Animation, self).add_to_scene(ren)
        [ren.add(static_act) for static_act in self._static_actors]
        [ren.add(animation) for animation in self._animations]

        if self._motion_path_actor:
            ren.add(self._motion_path_actor)
        self._scene = ren
        self._added_to_scene = True
        self._added_to_scene_time = time.perf_counter()
        self.update_animation(0)
