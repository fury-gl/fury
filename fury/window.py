# -*- coding: utf-8 -*-
import gzip
import time
from tempfile import TemporaryDirectory as InTemporaryDirectory
from threading import Lock
from warnings import warn

import numpy as np
from scipy import ndimage

import fury.animation as anim
from fury import __version__ as fury_version
from fury.interactor import CustomInteractorStyle
from fury.io import load_image, save_image
from fury.lib import (
    Actor2D,
    Command,
    InteractorEventRecorder,
    InteractorStyleImage,
    InteractorStyleTrackballCamera,
    OpenGLRenderer,
    RenderLargeImage,
    RenderWindow,
    RenderWindowInteractor,
    Skybox,
    Volume,
    WindowToImageFilter,
    RenderPassCollection,
    DefaultRenderPass,
    SequencePass,
    SSAAPass,
    CameraPass,
    colors,
    numpy_support,
)
from fury.shaders.base import GL_NUMBERS as _GL
from fury.utils import asbytes

try:
    basestring
except NameError:
    basestring = str


class Scene(OpenGLRenderer):
    """Your scene class.

    This is an important object that is responsible for preparing objects
    e.g. actors and volumes for rendering. This is a more pythonic version
    of ``vtkRenderer`` providing simple methods for adding and removing actors
    but also it provides access to all the functionality
    available in ``vtkRenderer`` if necessary.
    """

    def __init__(self, background=(0, 0, 0), skybox=None):
        self.__skybox = skybox
        self.__skybox_actor = None
        if skybox:
            self.AutomaticLightCreationOff()
            self.UseImageBasedLightingOn()
            self.UseSphericalHarmonicsOff()
            self.SetEnvironmentTexture(self.__skybox)
            self.skybox()

    def background(self, color):
        """Set a background color."""
        self.SetBackground(color)

    def skybox(self, visible=True, gamma_correct=True):
        """Show or hide the skybox.

        Parameters
        ----------
        visible : bool
            Whether to show the skybox or not.
        gamma_correct : bool
            Whether to apply gamma correction to the skybox or not.

        """
        if self.__skybox:
            if visible:
                if self.__skybox_actor is None:
                    self.__skybox_actor = Skybox()
                    self.__skybox_actor.SetTexture(self.__skybox)
                    if gamma_correct:
                        self.__skybox_actor.GammaCorrectOn()
                self.add(self.__skybox_actor)
            else:
                self.rm(self.__skybox_actor)
        else:
            warn('Scene created without a skybox. Nothing to show or hide.')

    def add(self, *actors):
        """Add an actor to the scene."""
        for actor in actors:
            if isinstance(actor, Volume):
                self.AddVolume(actor)
            elif isinstance(actor, Actor2D):
                self.AddActor2D(actor)
            elif hasattr(actor, 'add_to_scene'):
                actor.add_to_scene(self)
            else:
                self.AddActor(actor)

    def rm(self, *actors):
        """Remove more than once actors at once."""
        for actor in actors:
            self.RemoveActor(actor)

    def clear(self):
        """Remove all actors from the scene."""
        self.RemoveAllViewProps()

    def rm_all(self):
        """Remove all actors from the scene."""
        self.RemoveAllViewProps()

    def projection(self, proj_type='perspective'):
        """Decide between parallel or perspective projection.

        Parameters
        ----------
        proj_type : str
            Can be 'parallel' or 'perspective' (default).

        """
        if proj_type == 'parallel':
            self.GetActiveCamera().ParallelProjectionOn()
        else:
            self.GetActiveCamera().ParallelProjectionOff()

    def reset_camera(self):
        """Reset the camera to an automatic position given by the engine."""
        self.ResetCamera()

    def reset_camera_tight(self, margin_factor=1.02):
        """Resets camera so the content fit tightly within the window.

        Parameters
        ----------
        margin_factor : float (optional)
            Margin added around the content. Default: 1.02.

        """
        self.ComputeAspect()
        cam = self.GetActiveCamera()
        aspect = self.GetAspect()

        X1, X2, Y1, Y2, Z1, Z2 = self.ComputeVisiblePropBounds()
        width, height = X2 - X1, Y2 - Y1
        center = np.array((X1 + width / 2.0, Y1 + height / 2.0, 0))

        angle = np.pi * cam.GetViewAngle() / 180.0
        dist = max(width / aspect[0], height) / np.sin(angle / 2.0) / 2.0
        position = center + np.array((0, 0, dist * margin_factor))

        cam.SetViewUp(0, 1, 0)
        cam.SetPosition(*position)
        cam.SetFocalPoint(*center)
        self.ResetCameraClippingRange(X1, X2, Y1, Y2, Z1, Z2)

        parallelScale = max(width / aspect[0], height) / 2.0
        cam.SetParallelScale(parallelScale * margin_factor)

    def reset_clipping_range(self):
        """Reset the camera to an automatic position given by the engine."""
        self.ResetCameraClippingRange()

    def camera(self):
        """Return the camera object."""
        return self.GetActiveCamera()

    def get_camera(self):
        """Return Camera information: Position, Focal Point, View Up."""
        cam = self.GetActiveCamera()
        return cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()

    def camera_info(self):
        """Return Camera information."""
        cam = self.camera()
        print('# Active Camera')
        print('   Position (%.2f, %.2f, %.2f)' % cam.GetPosition())
        print('   Focal Point (%.2f, %.2f, %.2f)' % cam.GetFocalPoint())
        print('   View Up (%.2f, %.2f, %.2f)' % cam.GetViewUp())

    def set_camera(self, position=None, focal_point=None, view_up=None):
        """Set up camera position / Focal Point / View Up."""
        if position is not None:
            self.GetActiveCamera().SetPosition(*position)
        if focal_point is not None:
            self.GetActiveCamera().SetFocalPoint(*focal_point)
        if view_up is not None:
            self.GetActiveCamera().SetViewUp(*view_up)
        self.ResetCameraClippingRange()

    def size(self):
        """Scene size."""
        return self.GetSize()

    def zoom(self, value):
        """Rescale scene's camera.

        In perspective mode, decrease the view angle by the specified
        factor. In parallel mode, decrease the parallel scale by the specified
        factor. A value greater than 1 is a zoom-in, a value less than 1 is a
        zoom-out.

        """
        self.GetActiveCamera().Zoom(value)

    def azimuth(self, angle):
        """Rotate scene's camera.

        Rotate the camera about the view up vector centered at the focal
        point. Note that the view up vector is whatever was set via SetViewUp,
        and is not necessarily perpendicular to the direction of projection.
        The result is a horizontal rotation of the camera.

        """
        self.GetActiveCamera().Azimuth(angle)

    def yaw(self, angle):
        """Yaw scene's camera.

        Rotate the focal point about the view up vector, using the camera's
        position as the center of rotation. Note that the view up vector is
        whatever was set via SetViewUp, and is not necessarily perpendicular
        to the direction of projection. The result is a horizontal rotation of
        the scene.

        """
        self.GetActiveCamera().Yaw(angle)

    def elevation(self, angle):
        """Elevate scene's camera.

        Rotate the camera about the cross product of the negative of the
        direction of projection and the view up vector, using the focal point
        as the center of rotation. The result is a vertical rotation of the
        scene.
        """
        self.GetActiveCamera().Elevation(angle)

    def pitch(self, angle):
        """Pitch scene's camera.

        Rotate the focal point about the cross product of the view up
        vector and the direction of projection, using the camera's position as
        the center of rotation. The result is a vertical rotation of the
        camera.
        """
        self.GetActiveCamera().Pitch(angle)

    def roll(self, angle):
        """Roll scene's camera.

        Rotate the camera about the direction of projection. This will
        spin the camera about its axis.
        """
        self.GetActiveCamera().Roll(angle)

    def dolly(self, value):
        """Dolly In/Out scene's camera.

        Divide the camera's distance from the focal point by the given
        dolly value. Use a value greater than one to dolly-in toward the focal
        point, and use a value less than one to dolly-out away from the focal
        point.
        """
        self.GetActiveCamera().Dolly(value)

    def camera_direction(self):
        """Get camera direction.

        Get the vector in the direction from the camera position to the
        focal point. This is usually the opposite of the ViewPlaneNormal, the
        vector perpendicular to the screen, unless the view is oblique.
        """
        return self.GetActiveCamera().GetDirectionOfProjection()

    @property
    def last_render_time(self):
        """Returns the last render time in seconds."""
        return self.GetLastRenderTimeInSeconds()

    def fxaa_on(self):
        self.SetUseFXAA(True)

    def fxaa_off(self):
        self.SetUseFXAA(False)

    def enable_ssaa(self):
        """Turn SSAA on. Uses render passes."""
        collection_pass = RenderPassCollection()
        collection_pass.AddItem(DefaultRenderPass())

        sequence_pass = SequencePass()
        sequence_pass.SetPasses(collection_pass)

        camera_pass = CameraPass()
        camera_pass.SetDelegatePass(sequence_pass)

        ssaa_pass = SSAAPass()
        ssaa_pass.SetDelegatePass(camera_pass)

        self.SetPass(ssaa_pass)

    def msaa(self):
        """Turn MSAA on. Uses VTK Render/Sequence Pass, and MSAA Pass."""
        # TODO: Add MSAA to Scene()


class ShowManager:
    """Class interface between the scene, the window and the interactor."""

    def __init__(
        self,
        scene=None,
        title='FURY',
        size=(300, 300),
        png_magnify=1,
        reset_camera=True,
        order_transparent=False,
        interactor_style='custom',
        stereo='off',
        multi_samples=8,
        max_peels=4,
        occlusion_ratio=0.0,
    ):
        """Manage the visualization pipeline.

        Parameters
        ----------
        scene : Scene() or vtkRenderer()
            The scene that holds all the actors.
        title : string
            A string for the window title bar.
        size : (int, int)
            ``(width, height)`` of the window. Default is (300, 300).
        png_magnify : int
            Number of times to magnify the screenshot. This can be used to save
            high resolution screenshots when pressing 's' inside the window.
        reset_camera : bool
            Default is True. You can change this option to False if you want to
            keep the camera as set before calling this function.
        order_transparent : bool
            True is useful when you want to order transparent
            actors according to their relative position to the camera. The
            default option which is False will order the actors according to
            the order of their addition to the Scene().
        interactor_style : str or vtkInteractorStyle
            If str then if 'trackball' then vtkInteractorStyleTrackballCamera()
            is used, if 'image' then vtkInteractorStyleImage() is used (no
            rotation) or if 'custom' then CustomInteractorStyle is used.
            Otherwise you can input your own interactor style.
        stereo: string
            Set the stereo type. Default is 'off'. Other types include:

            * 'opengl': OpenGL frame-sequential stereo. Referred to as
              'CrystalEyes' by VTK.
            * 'anaglyph': For use with red/blue glasses. See VTK docs to
              use different colors.
            * 'interlaced': Line interlaced.
            * 'checkerboard': Checkerboard interlaced.
            * 'left': Left eye only.
            * 'right': Right eye only.
            * 'horizontal': Side-by-side.

        multi_samples : int
            Number of samples for anti-aliazing (Default 8).
            For no anti-aliasing use 0.
        max_peels : int
            Maximum number of peels for depth peeling (Default 4).
        occlusion_ratio : float
            Occlusion ration for depth peeling (Default 0 - exact image).

        Attributes
        ----------
        scene : Scene() or vtkRenderer()
        iren : vtkRenderWindowInteractor()
        style : vtkInteractorStyle()
        window : vtkRenderWindow()

        Examples
        --------
        >>> from fury import actor, window
        >>> scene = window.Scene()
        >>> scene.add(actor.axes())
        >>> showm = window.ShowManager(scene)
        >>> # showm.render()
        >>> # showm.start()

        """
        if scene is None:
            scene = Scene()
        self.scene = scene
        self.title = title
        self.size = size
        self.png_magnify = png_magnify
        self.reset_camera = reset_camera
        self.order_transparent = order_transparent
        self.interactor_style = interactor_style
        self.stereo = stereo
        self.timers = []
        self.mutex = Lock()
        self._fps = 0
        self._last_render_time = 0

        if self.reset_camera:
            self.scene.ResetCamera()

        self.window = RenderWindow()

        if self.stereo.lower() != 'off':
            enable_stereo(self.window, self.stereo)

        self.window.AddRenderer(scene)

        self.window.SetSize(size[0], size[1])

        if self.order_transparent:
            occlusion_ratio = occlusion_ratio or 0.1
            antialiasing(
                self.scene,
                self.window,
                multi_samples=multi_samples,
                max_peels=max_peels,
                occlusion_ratio=occlusion_ratio,
            )

        if self.interactor_style == 'image':
            self.style = InteractorStyleImage()
        elif self.interactor_style == 'trackball':
            self.style = InteractorStyleTrackballCamera()
        elif self.interactor_style == 'custom':
            self.style = CustomInteractorStyle()
        else:
            self.style = interactor_style

        self.iren = RenderWindowInteractor()
        self.style.SetCurrentRenderer(self.scene)
        # Hack: below, we explicitly call the Python version of SetInteractor.
        self.style.SetInteractor(self.iren)
        self.iren.SetInteractorStyle(self.style)
        self.iren.SetRenderWindow(self.window)
        self._timelines = []
        self._animations = []
        self._animation_callback = None

    def initialize(self):
        """Initialize interaction."""
        self.iren.Initialize()

    @property
    def timelines(self):
        """Return a list of Timelines that were added to the ShowManager.

        Returns
        -------
        list[Timeline]:
            List of Timelines.

        """
        return self._timelines

    @property
    def animations(self):
        """Return a list of Animations that were added to the ShowManager.

        Returns
        -------
        list[Animation]:
            List of Animations.

        """
        return self._animations

    def add_animation(self, animation):
        """Add an Animation or a Timeline to the ShowManager.

        Adding an Animation or a Timeline to the ShowManager ensures that it
        gets added to the scene, gets updated and rendered without any extra
        code.

        Parameters
        ----------
        animation : Animation or Timeline
            The Animation or Timeline to be added to the ShowManager.

        """
        animation.add_to_scene(self.scene)
        if isinstance(animation, anim.Animation):
            if animation in self._animations:
                return
            self._animations.append(animation)
        elif isinstance(animation, anim.Timeline):
            if animation in self._timelines:
                return
            self._timelines.append(animation)

        if self._animation_callback is not None:
            return

        def animation_cbk(_obj, _event):
            [tl.update() for tl in self._timelines]
            [anim.update_animation() for anim in self._animations]
            self.render()

        self._animation_callback = self.add_timer_callback(True, 10, animation_cbk)

    def remove_animation(self, animation):
        """Remove an Animation or a Timeline from the ShowManager.

        Animation will be removed from the Scene as well as from the
        ShowManager.

        Parameters
        ----------
        animation : Animation or Timeline
            The Timeline to be removed.

        """
        if animation in self.timelines or animation in self.animations:
            animation.remove_from_scene(self.scene)
            if isinstance(animation, anim.Animation):
                self._animations.remove(animation)
            elif isinstance(animation, anim.Timeline):
                self._timelines.remove(animation)
            if not (len(self.timelines) or len(self.animations)):
                self.iren.DestroyTimer(self._animation_callback)
                self._animation_callback = None

    def render(self):
        """Render only once."""
        self.window.Render()
        # calculate the FPS
        self._fps = 1.0 / (time.perf_counter() - self._last_render_time)
        self._last_render_time = time.perf_counter()

    def is_done(self):
        """Check if show manager is done."""
        try:
            return self.iren.GetDone()
        except AttributeError:
            return True

    def start(self, multithreaded=False, desired_fps=60):
        """Start interaction.

        Parameters
        ----------
        multithreaded : bool
            Whether to use multithreading. (Default False)
        desired_fps : int
            Desired frames per second when using multithreading is enabled.
            (Default 60)

        """
        try:
            if self.title.upper() == 'FURY':
                self.window.SetWindowName(self.title + ' ' + fury_version)
            else:
                self.window.SetWindowName(self.title)
            if multithreaded:
                while self.iren.GetDone() is False:
                    start_time = time.perf_counter()
                    self.lock()
                    self.window.MakeCurrent()
                    self.iren.ProcessEvents()  # Check if we can really do that
                    self.window.Render()
                    release_context(self.window)
                    self.release_lock()
                    end_time = time.perf_counter()
                    # throttle to 60fps to avoid busy wait
                    time_per_frame = 1.0 / desired_fps
                    if end_time - start_time < time_per_frame:
                        time.sleep(time_per_frame - (end_time - start_time))
            else:
                self.render()
                self.iren.Start()

        except AttributeError:
            self.__init__(
                self.scene,
                self.title,
                size=self.size,
                png_magnify=self.png_magnify,
                reset_camera=self.reset_camera,
                order_transparent=self.order_transparent,
                interactor_style=self.interactor_style,
            )
            self.render()
            if self.title.upper() == 'FURY':
                self.window.SetWindowName(self.title + ' ' + fury_version)
            else:
                self.window.SetWindowName(self.title)
            self.iren.Start()

        self.window.RemoveRenderer(self.scene)
        self.scene.SetRenderWindow(None)
        self.window.Finalize()
        del self.iren
        del self.window

    def lock(self):
        """Lock the render window."""
        self.mutex.acquire()

    def release_lock(self):
        """Release the lock of the render window."""
        self.mutex.release()

    def lock_current(self):
        """Lock the render window and acquire the current context and
        check if the lock was successfully acquired.

        Returns
        -------
        successful : bool
        Returns if the lock was acquired.

        """
        if self.is_done():
            return False
        if not hasattr(self, 'window'):
            return False
        try:
            self.lock()
            self.window.MakeCurrent()
            return True
        except AttributeError:
            return False

    def release_current(self):
        """Release the window context and lock of the render window."""
        release_context(self.window)
        self.release_lock()

    def wait(self):
        """Wait for thread to finish."""
        if self.thread:
            self.thread.join()

    @property
    def frame_rate(self):
        """Returns number of frames per second."""
        return self._fps

    def record_events(self):
        """Record events during the interaction.

        The recording is represented as a list of VTK events that happened
        during the interaction. The recorded events are then returned.

        Returns
        -------
        events : str
            Recorded events (one per line).

        Notes
        -----
        Since VTK only allows recording events to a file, we use a
        temporary file from which we then read the events.

        """
        with InTemporaryDirectory():
            filename = 'recorded_events.log'
            recorder = InteractorEventRecorder()
            recorder.SetInteractor(self.iren)
            recorder.SetFileName(filename)

            def _stop_recording_and_close(_obj, _evt):
                if recorder:
                    recorder.Stop()
                self.iren.TerminateApp()

            self.iren.AddObserver('ExitEvent', _stop_recording_and_close)

            recorder.EnabledOn()
            recorder.Record()

            self.render()
            self.iren.Start()
            # Deleting this object is the unique way
            # to close the file.
            recorder = None
            # Retrieved recorded events.
            with open(filename, 'r') as f:
                events = f.read()
        return events

    def record_events_to_file(self, filename='record.log'):
        """Record events during the interaction.

        The recording is represented as a list of VTK events
        that happened during the interaction. The recording is
        going to be saved into `filename`.

        Parameters
        ----------
        filename : str
            Name of the file that will contain the recording (.log|.log.gz).

        """
        events = self.record_events()

        # Compress file if needed
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wb') as fgz:
                fgz.write(asbytes(events))
        else:
            with open(filename, 'w') as f:
                f.write(events)

    def play_events(self, events):
        """Play recorded events of a past interaction.

        The VTK events that happened during the recorded interaction will be
        played back.

        Parameters
        ----------
        events : str
            Recorded events (one per line).

        """
        recorder = InteractorEventRecorder()
        recorder.SetInteractor(self.iren)

        recorder.SetInputString(events)
        recorder.ReadFromInputStringOn()
        self.initialize()
        # self.render()
        recorder.Play()

        # Finalize seems very important otherwise
        # the recording window will not close.
        self.window.RemoveRenderer(self.scene)
        self.scene.SetRenderWindow(None)
        self.window.Finalize()
        self.exit()

        # print('After Finalize and Exit')

        # del self.iren
        # del self.window

    def play_events_from_file(self, filename):
        """Play recorded events of a past interaction.

        The VTK events that happened during the recorded interaction will be
        played back from `filename`.

        Parameters
        ----------
        filename : str
            Name of the file containing the recorded events (.log|.log.gz).

        """
        # Uncompress file if needed.
        if filename.endswith('.gz'):
            with gzip.open(filename, 'r') as f:
                events = f.read()
        else:
            with open(filename) as f:
                events = f.read()

        self.play_events(events)

    def add_window_callback(self, win_callback, event=Command.ModifiedEvent):
        """Add window callbacks."""
        self.window.AddObserver(event, win_callback)
        self.window.Render()

    def add_timer_callback(self, repeat, duration, timer_callback):
        if not self.iren.GetInitialized():
            self.initialize()
        self.iren.AddObserver('TimerEvent', timer_callback)

        if repeat:
            timer_id = self.iren.CreateRepeatingTimer(duration)
        else:
            timer_id = self.iren.CreateOneShotTimer(duration)
        self.timers.append(timer_id)
        return timer_id

    def add_iren_callback(self, iren_callback, event='MouseMoveEvent'):
        if not self.iren.GetInitialized():
            self.initialize()
        self.iren.AddObserver(event, iren_callback)

    def destroy_timer(self, timer_id):
        self.iren.DestroyTimer(timer_id)
        del self.timers[self.timers.index(timer_id)]

    def destroy_timers(self):
        for timer_id in self.timers:
            self.destroy_timer(timer_id)

    def exit(self):
        """Close window and terminate interactor."""
        # if is_osx and self.timers:
        # OSX seems to not destroy correctly timers
        # segfault 11 appears sometimes if we do not do it manually.
        # self.iren.GetRenderWindow().Finalize()
        self.iren.TerminateApp()
        self.destroy_timers()
        self.timers.clear()

    def save_screenshot(self, fname, magnification=1, size=None, stereo=None):
        """Save a screenshot of the current window in the specified filename.

        Parameters
        ----------
        fname : str or None
            File name where to save the screenshot.
        magnification : int, optional
            Applies a magnification factor to the scene before taking the
            screenshot which improves the quality. A value greater than 1
            increases the quality of the image. However, the output size will
            be larger. For example, 200x200 image with magnification of 2 will
            result in a 400x400 image. Default is 1.
        size : tuple of 2 ints, optional
            Size of the output image in pixels. If None, the size of the scene
            will be used. If magnification > 1, then the size will be
            determined by the magnification factor. Default is None.
        stereo : str, optional
            Set the type of stereo for the screenshot. Supported values are:

                * 'opengl': OpenGL frame-sequential stereo. Referred to as
                  'CrystalEyes' by VTK.
                * 'anaglyph': For use with red/blue glasses. See VTK docs to
                  use different colors.
                * 'interlaced': Line interlaced.
                * 'checkerboard': Checkerboard interlaced.
                * 'left': Left eye only.
                * 'right': Right eye only.
                * 'horizontal': Side-by-side.

        """
        if size is None:
            size = self.size
        if stereo is None:
            stereo = self.stereo.lower()

        record(
            scene=self.scene,
            out_path=fname,
            magnification=magnification,
            size=size,
            stereo=stereo,
        )


def show(
    scene,
    title='FURY',
    size=(300, 300),
    png_magnify=1,
    reset_camera=True,
    order_transparent=False,
    stereo='off',
    multi_samples=8,
    max_peels=4,
    occlusion_ratio=0.0,
):
    """Show window with current scene.

    Parameters
    ----------
    scene : Scene() or vtkRenderer()
        The scene that holds all the actors.
    title : string
        A string for the window title bar. Default is FURY and current version.
    size : (int, int)
        ``(width, height)`` of the window. Default is (300, 300).
    png_magnify : int
        Number of times to magnify the screenshot. Default is 1. This can be
        used to save high resolution screenshots when pressing 's' inside the
        window.
    reset_camera : bool
        Default is True. You can change this option to False if you want to
        keep the camera as set before calling this function.
    order_transparent : bool
        True is useful when you want to order transparent
        actors according to their relative position to the camera. The default
        option which is False will order the actors according to the order of
        their addition to the Scene().
    stereo : string
        Set the stereo type. Default is 'off'. Other types include:

        * 'opengl': OpenGL frame-sequential stereo. Referred to as
          'CrystalEyes' by VTK.
        * 'anaglyph': For use with red/blue glasses. See VTK docs to
          use different colors.
        * 'interlaced': Line interlaced.
        * 'checkerboard': Checkerboard interlaced.
        * 'left': Left eye only.
        * 'right': Right eye only.
        * 'horizontal': Side-by-side.

    multi_samples : int
        Number of samples for anti-aliazing (Default 8).
        For no anti-aliasing use 0.
    max_peels : int
        Maximum number of peels for depth peeling (Default 4).
    occlusion_ratio : float
        Occlusion ration for depth peeling (Default 0 - exact image).

    Examples
    --------
    >>> import numpy as np
    >>> from fury import window, actor
    >>> r = window.Scene()
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]
    >>> colors=np.array([[0.2,0.2,0.2],[0.8,0.8,0.8]])
    >>> c=actor.line(lines,colors)
    >>> r.add(c)
    >>> l=actor.label(text="Hello")
    >>> r.add(l)
    >>> #window.show(r)

    See Also
    --------
    fury.window.record
    fury.window.snapshot

    """
    show_manager = ShowManager(
        scene,
        title,
        size,
        png_magnify,
        reset_camera,
        order_transparent,
        stereo=stereo,
        multi_samples=multi_samples,
        max_peels=max_peels,
        occlusion_ratio=occlusion_ratio,
    )
    show_manager.render()
    show_manager.start()


def record(
    scene=None,
    cam_pos=None,
    cam_focal=None,
    cam_view=None,
    out_path=None,
    path_numbering=False,
    n_frames=1,
    az_ang=10,
    magnification=1,
    size=(300, 300),
    reset_camera=True,
    screen_clip=False,
    stereo='off',
    verbose=False,
):
    """Record a video of your scene.

    Records a video as a series of ``.png`` files of your scene by rotating the
    azimuth angle az_angle in every frame.

    Parameters
    ----------
    scene : Scene() or vtkRenderer() object
        Scene instance
    cam_pos : None or sequence (3,), optional
        Camera's position. If None then default camera's position is used.
    cam_focal : None or sequence (3,), optional
        Camera's focal point. If None then default camera's focal point is
        used.
    cam_view : None or sequence (3,), optional
        Camera's view up direction. If None then default camera's view up
        vector is used.
    out_path : str, optional
        Output path for the frames. If None a default fury.png is created.
    path_numbering : bool
        When recording it changes out_path to out_path + str(frame number)
    n_frames : int, optional
        Number of frames to save, default 1
    az_ang : float, optional
        Azimuthal angle of camera rotation.
    magnification : int, optional
        How much to magnify the saved frame. Default is 1. A value greater
        than 1 increases the quality of the image. However, the output
        size will be larger. For example, 200x200 image with magnification
        of 2 will be a 400x400 image.
    size : (int, int)
        ``(width, height)`` of the window. Default is (300, 300).
    screen_clip: bool
        Clip the png based on screen resolution. Default is False.
    reset_camera : bool
        If True Call ``scene.reset_camera()``. Otherwise you need to set the
         camera before calling this function.
    stereo: string
        Set the stereo type. Default is 'off'. Other types include:

        * 'opengl': OpenGL frame-sequential stereo. Referred to as
          'CrystalEyes' by VTK.
        * 'anaglyph': For use with red/blue glasses. See VTK docs to
          use different colors.
        * 'interlaced': Line interlaced.
        * 'checkerboard': Checkerboard interlaced.
        * 'left': Left eye only.
        * 'right': Right eye only.
        * 'horizontal': Side-by-side.

    verbose : bool
        print information about the camera. Default is False.

    Examples
    --------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> a = actor.axes()
    >>> scene.add(a)
    >>> # uncomment below to record
    >>> # window.record(scene)
    >>> # check for new images in current directory

    """
    if scene is None:
        scene = Scene()

    renWin = RenderWindow()

    renWin.SetOffScreenRendering(1)
    renWin.SetBorders(screen_clip)
    renWin.AddRenderer(scene)
    renWin.SetSize(size[0], size[1])

    # scene.GetActiveCamera().Azimuth(180)

    if reset_camera:
        scene.ResetCamera()

    if stereo.lower() != 'off':
        enable_stereo(renWin, stereo)

    renderLarge = RenderLargeImage()
    renderLarge.SetInput(scene)
    renderLarge.SetMagnification(magnification)
    renderLarge.Update()

    ang = 0

    if cam_pos is not None:
        cx, cy, cz = cam_pos
        scene.GetActiveCamera().SetPosition(cx, cy, cz)
    if cam_focal is not None:
        fx, fy, fz = cam_focal
        scene.GetActiveCamera().SetFocalPoint(fx, fy, fz)
    if cam_view is not None:
        ux, uy, uz = cam_view
        scene.GetActiveCamera().SetViewUp(ux, uy, uz)

    cam = scene.GetActiveCamera()
    if verbose:
        print('Camera Position (%.2f, %.2f, %.2f)' % cam.GetPosition())
        print('Camera Focal Point (%.2f, %.2f, %.2f)' % cam.GetFocalPoint())
        print('Camera View Up (%.2f, %.2f, %.2f)' % cam.GetViewUp())

    for i in range(n_frames):
        scene.GetActiveCamera().Azimuth(ang)
        renderLarge = RenderLargeImage()
        renderLarge.SetInput(scene)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()

        if path_numbering:
            if out_path is None:
                filename = str(i).zfill(6) + '.png'
            else:
                filename = out_path + str(i).zfill(6) + '.png'
        else:
            if out_path is None:
                filename = 'fury.png'
            else:
                filename = out_path

        arr = numpy_support.vtk_to_numpy(
            renderLarge.GetOutput().GetPointData().GetScalars()
        )
        w, h, _ = renderLarge.GetOutput().GetDimensions()
        components = renderLarge.GetOutput().GetNumberOfScalarComponents()
        arr = arr.reshape((h, w, components))
        arr = np.flipud(arr)
        save_image(arr, filename)

        ang = +az_ang

    renWin.RemoveRenderer(scene)
    renWin.Finalize()


def antialiasing(scene, win, multi_samples=8, max_peels=4, occlusion_ratio=0.0):
    """Enable anti-aliasing and ordered transparency.

    Parameters
    ----------
    scene : Scene
    win : Window
        Provided by ShowManager.window attribute.
    multi_samples : int
        Number of samples for anti-aliasing (Default 8).
        For no anti-aliasing use 0.
    max_peels : int
        Maximum number of peels for depth peeling (Default 4).
    occlusion_ratio : float
        Occlusion ratio for depth peeling (Default 0 - exact image).

    """
    # Use a render window with alpha bits
    # as default is 0 (false))
    win.SetAlphaBitPlanes(True)

    # Force to not pick a framebuffer with a multisample buffer
    # (default is 8)
    win.SetMultiSamples(multi_samples)

    # TODO: enable these but test
    # win.SetBorders(True)
    # win.LineSmoothingOn(True)
    # win.PointSmoothingOn(True)
    # win.PolygonSmoothingOn(True)

    # Choose to use depth peeling (if supported)
    # (default is 0 (false)):
    scene.UseDepthPeelingOn()

    # Set depth peeling parameters
    # Set the maximum number of rendering passes (default is 4)
    scene.SetMaximumNumberOfPeels(max_peels)

    # Set the occlusion ratio (initial value is 0.0, exact image):
    scene.SetOcclusionRatio(occlusion_ratio)


def snapshot(
    scene,
    fname=None,
    size=(300, 300),
    offscreen=True,
    order_transparent=False,
    stereo='off',
    multi_samples=8,
    max_peels=4,
    occlusion_ratio=0.0,
    dpi=(72, 72),
    render_window=None,
):
    """Save a snapshot of the scene in a file or in memory.

    Parameters
    ----------
    scene : Scene() or vtkRenderer
        Scene instance
    fname : str or None
        Save PNG file. If None return only an array without saving PNG.
    size : (int, int)
        ``(width, height)`` of the window. Default is (300, 300).
    offscreen : bool
        Default True. Go stealth mode no window should appear.
    order_transparent : bool
        Default False. Use depth peeling to sort transparent objects.
        If True also enables anti-aliasing.

    stereo: string
        Set the stereo type. Default is 'off'. Other types include:

        * 'opengl': OpenGL frame-sequential stereo. Referred to as
          'CrystalEyes' by VTK.
        * 'anaglyph': For use with red/blue glasses. See VTK docs to
          use different colors.
        * 'interlaced': Line interlaced.
        * 'checkerboard': Checkerboard interlaced.
        * 'left': Left eye only.
        * 'right': Right eye only.
        * 'horizontal': Side-by-side.

    multi_samples : int
        Number of samples for anti-aliazing (Default 8).
        For no anti-aliasing use 0.
    max_peels : int
        Maximum number of peels for depth peeling (Default 4).
    occlusion_ratio : float
        Occlusion ration for depth peeling (Default 0 - exact image).
    dpi : float or (float, float)
        Dots per inch (dpi) for saved image.
        Single values are applied as dpi for both dimensions.
    render_window : RenderWindow
        If provided, use this window instead of creating a new one.

    Returns
    -------
    arr : ndarray
        Color array of size (width, height, 3) where the last dimension
        holds the RGB values.

    """
    width, height = size
    if render_window is None:
        render_window = RenderWindow()
        if offscreen:
            render_window.SetOffScreenRendering(1)
        if stereo.lower() != 'off':
            enable_stereo(render_window, stereo)
        render_window.AddRenderer(scene)
        render_window.SetSize(width, height)

        if order_transparent:
            antialiasing(
                scene, render_window, multi_samples, max_peels, occlusion_ratio
            )
        render_window.Render()

    window_to_image_filter = WindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components).copy()
    arr = np.flipud(arr)

    if fname is None:
        return arr

    save_image(arr, fname, dpi=dpi)

    render_window.RemoveRenderer(scene)
    render_window.Finalize()

    return arr


def analyze_scene(scene):
    class ReportScene:
        bg_color = None
        collection = None
        actors = None
        actors_classnames = None

    report = ReportScene()

    report.bg_color = scene.GetBackground()
    report.collection = scene.GetActors()
    report.actors = report.collection.GetNumberOfItems()

    report.collection.InitTraversal()
    report.actors_classnames = []
    for _ in range(report.actors):
        class_name = report.collection.GetNextActor().GetClassName()
        report.actors_classnames.append(class_name)

    return report


def analyze_snapshot(
    im, bg_color=colors.black, colors=None, find_objects=True, strel=None
):
    """Analyze snapshot from memory or file.

    Parameters
    ----------
    im: str or array
        If string then the image is read from a file otherwise the image is
        read from a numpy array. The array is expected to be of shape (X, Y, 3)
        or (X, Y, 4) where the last dimensions are the RGB or RGBA values.
    colors: tuple (3,) or list of tuples (3,)
        List of colors to search in the image
    find_objects: bool
        If True it will calculate the number of objects that are different
        from the background and return their position in a new image.
    strel: 2d array
        Structure element to use for finding the objects.

    Returns
    -------
    report : ReportSnapshot
        This is an object with attributes like ``colors_found`` that give
        information about what was found in the current snapshot array ``im``.

    """
    if isinstance(im, basestring):
        im = load_image(im)

    class ReportSnapshot:
        objects = None
        labels = None
        colors_found = False

        def __str__(self):
            msg = 'Report:\n-------\n'
            msg += 'objects: {}\n'.format(self.objects)
            msg += 'labels: \n{}\n'.format(self.labels)
            msg += 'colors_found: {}\n'.format(self.colors_found)
            return msg

    report = ReportSnapshot()

    if colors is not None:
        if isinstance(colors, tuple):
            colors = [colors]
        flags = [False] * len(colors)
        for (i, col) in enumerate(colors):
            # find if the current color exist in the array
            flags[i] = np.any(np.any(np.all(np.equal(im[..., :3], col[:3]), axis=-1)))

        report.colors_found = flags

    if find_objects is True:
        weights = [0.299, 0.587, 0.144]
        gray = np.dot(im[..., :3], weights)
        bg_color2 = im[0, 0]
        background = np.dot(bg_color2, weights)

        if strel is None:
            strel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        labels, objects = ndimage.label(gray != background, strel)
        report.labels = labels
        report.objects = objects

    return report


def enable_stereo(renwin, stereo_type):
    """Enable the given stereo type on the RenderWindow.

    Parameters
    ----------
    renwin: vtkRenderWindow
    stereo_type: string
        * 'opengl': OpenGL frame-sequential stereo. Referred to as
          'CrystalEyes' by VTK.
        * 'anaglyph': For use with red/blue glasses. See VTK docs to
          use different colors.
        * 'interlaced': Line interlaced.
        * 'checkerboard': Checkerboard interlaced.
        * 'left': Left eye only.
        * 'right': Right eye only.
        * 'horizontal': Side-by-side.

    """
    renwin.GetStereoCapableWindow()
    renwin.StereoCapableWindowOn()
    renwin.StereoRenderOn()

    stereo_type = stereo_type.lower()

    # stereo type ints from
    # https://gitlab.kitware.com/vtk/vtk/blob/master/Rendering/Core/vtkRenderWindow.h#L57
    stereo_type_dictionary = {
        'opengl': 1,
        'interlaced': 3,
        'anaglyph': 7,
        'checkerboard': 8,
        'horizontal': 9,
    }

    # default to horizontal since it is easy to see if it is working
    if stereo_type not in stereo_type_dictionary:
        warn('Unknown stereo type provided. ' "Setting stereo type to 'horizontal'.")
        stereo_type = 'horizontal'

    renwin.SetStereoType(stereo_type_dictionary[stereo_type])


def gl_get_current_state(gl_state):
    """Returns a dict which describes the current state of the opengl
    context

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    state_description = {
        glName: gl_state.GetEnumState(glNumber) for glName, glNumber in _GL.items()
    }
    return state_description


def gl_reset_blend(gl_state):
    """Redefines the state of the OpenGL context related with how the RGBA
    channels will be combined.

    Parameters
    ----------
    gl_state : vtkOpenGLState

    See more
    ---------
    [1] https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBlendEquation.xhtml
    [2] https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBlendFunc.xhtml
    vtk specification:
    [3] https://gitlab.kitware.com/vtk/vtk/-/blob/master/Rendering/OpenGL2/vtkOpenGLState.cxx#L1705

    """  # noqa
    gl_state.ResetGLBlendEquationState()
    gl_state.ResetGLBlendFuncState()


def gl_enable_depth(gl_state):
    """Enable OpenGl depth test

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_state.vtkglEnable(_GL['GL_DEPTH_TEST'])


def gl_disable_depth(gl_state):
    """Disable OpenGl depth test

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_state.vtkglDisable(_GL['GL_DEPTH_TEST'])


def gl_enable_blend(gl_state):
    """Enable OpenGl blending

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_state.vtkglEnable(_GL['GL_BLEND'])


def gl_disable_blend(gl_state):
    """This it will disable any gl behavior which has no
    function for opaque objects. This has the benefit of
    speeding up the rendering of the image.

    Parameters
    ----------
    gl_state : vtkOpenGLState

    See more
    --------
    [1] https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glFrontFace.xhtml

    """  # noqa

    gl_state.vtkglDisable(_GL['GL_CULL_FACE'])
    gl_state.vtkglDisable(_GL['GL_BLEND'])


def gl_set_additive_blending(gl_state):
    """Enable additive blending

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_reset_blend(gl_state)
    gl_state.vtkglEnable(_GL['GL_BLEND'])
    gl_state.vtkglDisable(_GL['GL_DEPTH_TEST'])
    gl_state.vtkglBlendFunc(_GL['GL_SRC_ALPHA'], _GL['GL_ONE'])


def gl_set_additive_blending_white_background(gl_state):
    """Enable additive blending for a white background

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_reset_blend(gl_state)
    gl_state.vtkglEnable(_GL['GL_BLEND'])
    gl_state.vtkglDisable(_GL['GL_DEPTH_TEST'])
    gl_state.vtkglBlendFuncSeparate(
        _GL['GL_SRC_ALPHA'],
        _GL['GL_ONE_MINUS_SRC_ALPHA'],
        _GL['GL_ONE'],
        _GL['GL_ZERO'],
    )


def gl_set_normal_blending(gl_state):
    """Enable normal blending

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_state.vtkglEnable(_GL['GL_BLEND'])
    gl_state.vtkglEnable(_GL['GL_DEPTH_TEST'])
    gl_state.vtkglBlendFunc(_GL['GL_ONE'], _GL['GL_ONE'])
    gl_state.vtkglBlendFuncSeparate(
        _GL['GL_SRC_ALPHA'],
        _GL['GL_ONE_MINUS_SRC_ALPHA'],
        _GL['GL_ONE'],
        _GL['GL_ONE_MINUS_SRC_ALPHA'],
    )


def gl_set_multiplicative_blending(gl_state):
    """Enable multiplicative blending

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_reset_blend(gl_state)
    gl_state.vtkglBlendFunc(_GL['GL_ZERO'], _GL['GL_SRC_COLOR'])


def gl_set_subtractive_blending(gl_state):
    """Enable subtractive blending

    Parameters
    ----------
    gl_state : vtkOpenGLState

    """
    gl_reset_blend(gl_state)
    gl_state.vtkglBlendFunc(_GL['GL_ZERO'], _GL['GL_ONE_MINUS_SRC_COLOR'])


def release_context(window):
    """Release the context of the window

    Parameters
    ----------
    window : vtkRenderWindow

    """
    window.ReleaseCurrent()
