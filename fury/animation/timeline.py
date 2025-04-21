"""Timeline class for keyframe animation."""

import os
from time import perf_counter

from PIL import Image
import numpy as np

from fury import window
from fury.animation.animation import Animation
from fury.decorators import warn_on_args_to_kwargs

# from fury.lib import RenderWindow, WindowToImageFilter, numpy_support
# from fury.ui.elements import PlaybackPanel

RenderWindow, WindowToImageFilter, numpy_support = None, None, None
PlaybackPanel = None


class Timeline:
    """Keyframe animation Timeline.

    Timeline is responsible for handling the playback of keyframes animations.
    It has multiple playback options which makes it easy to control the playback,
    speed, state of the animation with/without a GUI playback panel.

    Parameters
    ----------
    animations : Animation or list[Animation], optional
        Actor/s to be animated directly by the Timeline (main Animation).
        Default is None.
    playback_panel : bool, optional
        If True, the timeline will have a playback panel set, which can be used
        to control the playback of the timeline. Default is False.
    loop : bool, optional
        Whether to loop playing the timeline or play once. Default is True.
    length : float or int, optional
        The fixed length of the timeline. If set to None, the timeline will get
        its length from the animations that it controls automatically.
        Default is None.

    Attributes
    ----------
    playback_panel : PlaybackPanel or None
        The panel used to control the playback of the timeline.
    duration : float
        The duration of the timeline in seconds.
    current_timestamp : float
        The current time position in the timeline.
    speed : float
        The playback speed multiplier.
    playing : bool
        Whether the timeline is currently playing.
    stopped : bool
        Whether the timeline is currently stopped.
    paused : bool
        Whether the timeline is currently paused.
    loop : bool
        Whether the timeline is set to loop or play once.
    animations : list
        List of animations controlled by the timeline.
    has_playback_panel : bool
        Whether the timeline has a playback panel.
    """

    @warn_on_args_to_kwargs()
    def __init__(
        self, *, animations=None, playback_panel=False, loop=True, length=None
    ):
        """Initialize the Timeline.

        Parameters
        ----------
        animations : Animation or list[Animation], optional
            Actor/s to be animated directly by the Timeline (main Animation).
            Default is None.
        playback_panel : bool, optional
            If True, the timeline will have a playback panel set, which can be used
            to control the playback of the timeline. Default is False.
        loop : bool, optional
            Whether to loop playing the timeline or play once. Default is True.
        length : float or int, optional
            The fixed length of the timeline. If set to None, the timeline will get
            its length from the animations that it controls automatically.
            Default is None.
        """
        self._scene = None
        self.playback_panel = None
        self._current_timestamp = 0
        self._speed = 1.0
        self._last_started_time = 0
        self._playing = False
        self._animations = []
        self._loop = loop
        self._length = length
        self._duration = length if length is not None else 0.0

        if playback_panel:

            def set_loop(is_loop):
                self._loop = is_loop

            def set_speed(speed):
                self.speed = speed

            self.playback_panel = PlaybackPanel(loop=self._loop)
            self.playback_panel.on_play = self.play
            self.playback_panel.on_stop = self.stop
            self.playback_panel.on_pause = self.pause
            self.playback_panel.on_loop_toggle = set_loop
            self.playback_panel.on_progress_bar_changed = self.seek
            self.playback_panel.on_speed_changed = set_speed

        if animations is not None:
            self.add_animation(animations)

    def update_duration(self):
        """Update and return the duration of the Timeline.

        Calculate the duration based on either a fixed length or the maximum
        duration of all animations controlled by this Timeline.

        Returns
        -------
        float
            The duration of the Timeline in seconds.
        """
        if self._length is not None:
            self._duration = self._length
        else:
            self._duration = max(
                [0.0] + [anim.update_duration() for anim in self._animations]
            )
        if self.has_playback_panel:
            self.playback_panel.final_time = self.duration
        return self.duration

    @property
    def duration(self):
        """Return the duration of the Timeline.

        Returns
        -------
        float
            The duration of the Timeline in seconds.
        """
        return self._duration

    def play(self):
        """Play the animation.

        Start playing the timeline from the current timestamp. If the current timestamp
        is at the end of the timeline, it will reset to the beginning.
        """
        if not self.playing:
            if self.current_timestamp >= self.duration:
                self.current_timestamp = 0
            self._last_started_time = (
                perf_counter() - self._current_timestamp / self.speed
            )
            self._playing = True

    def pause(self):
        """Pause the animation.

        Freeze the animation at its current timestamp.
        """
        self._current_timestamp = self.current_timestamp
        self._playing = False

    def stop(self):
        """Stop the animation.

        Reset the timeline to the beginning and stop playback.
        """
        self._current_timestamp = 0
        self._playing = False
        self.update(force=True)

    def restart(self):
        """Restart the animation.

        Reset the timeline to the beginning and start playing.
        """
        self._current_timestamp = 0
        self._playing = True
        self.update(force=True)

    @property
    def current_timestamp(self):
        """Get current timestamp of the Timeline.

        Returns
        -------
        float
            The current position in seconds of the Timeline.
        """
        if self.playing:
            self._current_timestamp = (
                perf_counter() - self._last_started_time
            ) * self.speed
        return self._current_timestamp

    @current_timestamp.setter
    def current_timestamp(self, timestamp):
        """Set the current timestamp of the Timeline.

        Parameters
        ----------
        timestamp : float
            The time to set as current position of the Timeline in seconds.
        """
        self.seek(timestamp)

    def seek(self, timestamp):
        """Set the current timestamp of the Timeline.

        Parameters
        ----------
        timestamp : float
            The time in seconds to seek to. Will be clamped between 0 and the
            timeline duration.
        """
        # assuring timestamp value is in the timeline range
        if timestamp < 0:
            timestamp = 0
        elif timestamp > self.duration:
            timestamp = self.duration
        if self.playing:
            self._last_started_time = perf_counter() - timestamp / self.speed
        else:
            self._current_timestamp = timestamp
            self.update(force=True)

    def seek_percent(self, percent):
        """Seek to a percentage of the Timeline's duration.

        Parameters
        ----------
        percent : float
            Percentage value from 0 to 100 of the timeline duration to seek to.
        """
        t = percent * self.duration / 100
        self.seek(t)

    @property
    def playing(self):
        """Return whether the Timeline is playing.

        Returns
        -------
        bool
            True if the Timeline is currently playing, False otherwise.
        """
        return self._playing

    @property
    def stopped(self):
        """Return whether the Timeline is stopped.

        Returns
        -------
        bool
            True if the Timeline is stopped (not playing and at timestamp 0),
            False otherwise.
        """
        return not self.playing and not self._current_timestamp

    @property
    def paused(self):
        """Return whether the Timeline is paused.

        Returns
        -------
        bool
            True if the Timeline is paused (not playing but at a non-zero timestamp),
            False otherwise.
        """
        return not self.playing and self._current_timestamp is not None

    @property
    def speed(self):
        """Return the speed of the timeline's playback.

        Returns
        -------
        float
            The playback speed multiplier.
        """
        return self._speed

    @speed.setter
    def speed(self, speed):
        """Set the speed of the timeline's playback.

        Parameters
        ----------
        speed : float
            The playback speed multiplier. Values greater than 1 speed up playback,
            values between 0 and 1 slow it down. Values of 0 or less are ignored.
        """
        current = self.current_timestamp
        if speed <= 0:
            return
        self._speed = speed
        self._last_started_time = perf_counter()
        self.current_timestamp = current

    @property
    def loop(self):
        """Return the loop setting of the timeline.

        Returns
        -------
        bool
            True if the timeline is set to loop playback, False if set to play once.
        """
        return self._loop

    @loop.setter
    def loop(self, loop):
        """Set the timeline's playback to loop or play once.

        Parameters
        ----------
        loop : bool
            When True, playback will loop continuously.
            When False, playback will stop at the end of the timeline.
        """
        self._loop = loop

    @property
    def has_playback_panel(self):
        """Return whether the Timeline has a playback panel.

        Returns
        -------
        bool
            True if the Timeline has a playback panel, False otherwise.
        """
        return self.playback_panel is not None

    @warn_on_args_to_kwargs()
    def record(
        self,
        *,
        fname=None,
        fps=30,
        speed=1.0,
        size=(900, 768),
        order_transparent=True,
        multi_samples=8,
        max_peels=4,
        show_panel=False,
    ):
        """Record the animation to a file or return frames.

        Parameters
        ----------
        fname : str, optional
            The output filename. Creates a GIF if name ends with '.gif', or an MP4
            video if name ends with '.mp4'. If None, only returns the frames array.
            Default is None.
        fps : int, optional
            The number of frames per second to record. Default is 30.
        speed : float, optional
            The playback speed multiplier for the recording. Default is 1.0.
        size : tuple of int, optional
            The dimensions of the recording as (width, height). Default is (900, 768).
        order_transparent : bool, optional
            Whether to use depth peeling to sort transparent objects. If True,
            also enables anti-aliasing. Default is True.
        multi_samples : int, optional
            Number of samples for anti-aliasing. Use 0 for no anti-aliasing.
            Default is 8.
        max_peels : int, optional
            Maximum number of peels for depth peeling. Default is 4.
        show_panel : bool, optional
            Controls whether to show the playback panel (if True) or hide it
            (if False) in the recording. Default is False.

        Returns
        -------
        list
            List of frames as PIL Image objects.

        Notes
        -----
        It's recommended to use 30 or 50 FPS when recording to a GIF file.
        To save as MP4, OpenCV must be installed.
        """
        ext = os.path.splitext(fname)[-1]

        mp4 = ext == ".mp4"

        if mp4:
            try:
                import cv2
            except ImportError as err:
                raise ImportError(
                    "OpenCV must be installed in order to " "save as MP4 video."
                ) from err
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(fname, fourcc, fps, size)

        duration = self.duration
        step = speed / fps
        frames = []
        t = 0
        scene = self._scene
        if not scene:
            scene = window.Scene()
            scene.add(self)

        _hide_panel = False
        if self.has_playback_panel and not show_panel:
            self.playback_panel.hide()
            _hide_panel = True
        render_window = RenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(scene)
        render_window.SetSize(*size)

        if order_transparent:
            window.antialiasing(scene, render_window, multi_samples, max_peels, 0)

        render_window = RenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(scene)
        render_window.SetSize(*size)

        if order_transparent:
            window.antialiasing(scene, render_window, multi_samples, max_peels, 0)

        window_to_image_filter = WindowToImageFilter()

        print("Recording...")
        while t < duration:
            self.seek(t)
            render_window.Render()
            window_to_image_filter.SetInput(render_window)
            window_to_image_filter.Update()
            window_to_image_filter.Modified()
            vtk_image = window_to_image_filter.GetOutput()
            h, w, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            snap = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components)
            corrected_snap = np.flipud(snap)

            if mp4:
                cv_img = cv2.cvtColor(corrected_snap, cv2.COLOR_RGB2BGR)
                out.write(cv_img)
            else:
                pillow_snap = Image.fromarray(corrected_snap)
                frames.append(pillow_snap)

            t += step

        print("Saving...")

        if fname is None:
            return frames

        if mp4:
            out.release()
        else:
            frames[0].save(
                fname,
                append_images=frames[1:],
                loop=0,
                duration=1000 / fps,
                save_all=True,
            )

        if _hide_panel:
            self.playback_panel.show()

        return frames

    def add_animation(self, animation):
        """Add Animation or list of Animations to the Timeline.

        Parameters
        ----------
        animation : Animation or list[Animation] or tuple[Animation]
            Animation object(s) to be added to this Timeline.

        Raises
        ------
        TypeError
            If the provided animation is not an Animation object or a collection
            of Animation objects.
        """
        if isinstance(animation, (list, tuple)):
            [self.add_animation(anim) for anim in animation]
        elif isinstance(animation, Animation):
            animation._timeline = self
            self._animations.append(animation)
            self.update_duration()
        else:
            raise TypeError("Expected an Animation, a list or a tuple.")

    @property
    def animations(self) -> "list[Animation]":
        """Return all animations controlled by this Timeline.

        Returns
        -------
        list
            List of Animation objects controlled by the timeline.
        """
        return self._animations

    @warn_on_args_to_kwargs()
    def update(self, *, force=False):
        """Update the timeline and all controlled animations.

        Updates the Timeline and all animations it controls. Also updates the
        playback panel if one exists.

        Parameters
        ----------
        force : bool, optional
            If True, the timeline will update even when paused or stopped,
            which may use more resources. Default is False.
        """
        time = self.current_timestamp
        if self.has_playback_panel:
            self.playback_panel.current_time = time
        if time > self.duration:
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
            [anim.update_animation(time=time) for anim in self._animations]

    def add_to_scene(self, scene):
        """Add Timeline and all of its Animations to the scene.

        Parameters
        ----------
        scene : Scene
            The scene to add the Timeline and its Animations to.
        """
        self._scene = scene
        if self.has_playback_panel:
            self.playback_panel.add_to_scene(scene)
        [animation.add_to_scene(scene) for animation in self._animations]

    def remove_from_scene(self, scene):
        """Remove Timeline and all of its Animations from the scene.

        Parameters
        ----------
        scene : Scene
            The scene from which to remove the Timeline and its Animations.
        """
        self._scene = None
        if self.has_playback_panel:
            scene.rm(*tuple(self.playback_panel.actors))
        [animation.remove_from_scene(scene) for animation in self._animations]
