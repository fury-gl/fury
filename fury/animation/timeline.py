import time
from fury.ui.elements import PlaybackPanel
from fury.animation.animation import Animation


class Timeline(Animation):
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
        Whether loop playing the animation or not
    """

    def __init__(self, actors=None, playback_panel=False, length=None,
                 loop=False, motion_path_res=None):

        super().__init__(actors=actors, length=length, loop=loop,
                         motion_path_res=motion_path_res)
        self.playback_panel = None
        self._current_timestamp = 0
        self._speed = 1
        self._last_started_time = 0
        self._playing = False
        self._length = length
        self._reverse_playing = True
        self._animations = []
        self._timeline = None
        self._parent_animation = None
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

    def play(self):
        """Play the animation"""
        if not self.playing:
            if self.current_timestamp >= self.duration:
                self.current_timestamp = 0
            self._last_started_time = \
                time.perf_counter() - self._current_timestamp / self.speed
            self._playing = True

    def pause(self):
        """Pause the animation"""
        self._current_timestamp = self.current_timestamp
        self._playing = False

    def stop(self):
        """Stop the animation"""
        self._current_timestamp = 0
        self._playing = False
        self.update_animation(0)

    def restart(self):
        """Restart the animation"""
        self._current_timestamp = 0
        self._playing = True
        self.update_animation(0)

    @property
    def current_timestamp(self):
        """Get current timestamp of the Timeline.

        Returns
        -------
        float
            The current time of the Timeline.

        """
        if self.playing:
            self._current_timestamp = (time.perf_counter() -
                                       self._last_started_time) * self.speed
        return self._current_timestamp

    @current_timestamp.setter
    def current_timestamp(self, timestamp):
        """Set the current timestamp of the Timeline.

        Parameters
        ----------
        timestamp: float
            The time to set as current time of the Timeline.

        """
        self.seek(timestamp)

    def get_current_timestamp(self):
        """Get last calculated current timestamp of the Timeline.

        Returns
        -------
        float
            The last calculated current time of the Timeline.

        """
        return self._current_timestamp

    def seek(self, timestamp):
        """Set the current timestamp of the Timeline.

        Parameters
        ----------
        timestamp: float
            The time to seek.

        """
        # assuring timestamp value is in the timeline range
        if timestamp < 0:
            timestamp = 0
        elif timestamp > self.duration:
            timestamp = self.duration
        if self.playing:
            self._last_started_time = \
                time.perf_counter() - timestamp / self.speed
        else:
            self._current_timestamp = timestamp
            self.update_animation(timestamp)

    def seek_percent(self, percent):
        """Seek a percentage of the Timeline's final timestamp.

        Parameters
        ----------
        percent: float
            Value from 1 to 100.

        """
        t = percent * self.duration / 100
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
        """Set the playing state of the Timeline.

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
        return not self.playing and not self._current_timestamp

    @property
    def paused(self):
        """Return whether the Timeline is paused.

        Returns
        -------
        bool
            Timeline is paused if True.

        """

        return not self.playing and self._current_timestamp is not None

    @property
    def speed(self):
        """Return the speed of the timeline.

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
        """Return whether the `Timeline` has a playback panel.

        Returns
        -------
        bool: 'True' if the `Timeline` has a playback panel. otherwise, 'False'
        """
        return self.playback_panel is not None

    def add_child_animation(self, animation):
        """Add child Animation or list of Animations.

        Parameters
        ----------
        animation: Animation or list(Animation)
            Animation/s to be added.
        """
        super(Timeline, self).add_child_animation(animation)
        if isinstance(animation, Animation):
            animation._timeline = self
        self.update_duration()

    def update_duration(self):
        """Update and return the duration of the Timeline.

        Returns
        -------
        float
            The duration of the Timeline.
        """
        super(Timeline, self).update_duration()
        if self.has_playback_panel:
            self.playback_panel.final_time = self.duration
        return self.duration

    def update_animation(self, t=None):
        force = True
        if t is None:
            t = self.current_timestamp
            force = False
        if self.has_playback_panel:
            self.playback_panel.current_time = t
        if t > self.duration:
            if self._loop:
                self.seek(0)
            else:
                self.seek(self.duration)
                # Doing this will pause both the timeline and the panel.
                if self.has_playback_panel:
                    self.playback_panel.pause()
                else:
                    self.pause()
        if self.playing or force:
            super(Timeline, self).update_animation(t)
        if self._scene:
            self._scene.reset_clipping_range()
