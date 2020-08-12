# -*- coding: utf-8 -*-

import gzip
from warnings import warn

import numpy as np
from scipy import ndimage
import vtk
from vtk.util import numpy_support, colors

from tempfile import TemporaryDirectory as InTemporaryDirectory

from fury import __version__ as fury_version
from fury.decorators import is_osx
from fury.deprecator import deprecate_with_version
from fury.interactor import CustomInteractorStyle
from fury.io import load_image, save_image
from fury.utils import asbytes

try:
    basestring
except NameError:
    basestring = str


class Scene(vtk.vtkRenderer):
    """Your scene class.

    This is an important object that is responsible for preparing objects
    e.g. actors and volumes for rendering. This is a more pythonic version
    of ``vtkRenderer`` proving simple methods for adding and removing actors
    but also it provides access to all the functionality
    available in ``vtkRenderer`` if necessary.
    """

    def background(self, color):
        """Set a background color."""
        self.SetBackground(color)

    def add(self, *actors):
        """Add an actor to the scene."""
        for actor in actors:
            if isinstance(actor, vtk.vtkVolume):
                self.AddVolume(actor)
            elif isinstance(actor, vtk.vtkActor2D):
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
        """ Resets camera so the content fit tightly within the window.

        Parameters
        ----------
        margin_factor : float (optional)
            Margin added around the content. Default: 1.02.

        """
        self.ComputeAspect()
        cam = self.GetActiveCamera()
        aspect = self.GetAspect()

        X1, X2, Y1, Y2, Z1, Z2 = self.ComputeVisiblePropBounds()
        width, height = X2-X1, Y2-Y1
        center = np.array((X1 + width/2., Y1 + height/2., 0))

        angle = np.pi*cam.GetViewAngle()/180.
        dist = max(width/aspect[0], height) / np.sin(angle/2.) / 2.
        position = center + np.array((0, 0, dist*margin_factor))

        cam.SetViewUp(0, 1, 0)
        cam.SetPosition(*position)
        cam.SetFocalPoint(*center)
        self.ResetCameraClippingRange(X1, X2, Y1, Y2, Z1, Z2)

        parallelScale = max(width/aspect[0], height) / 2.
        cam.SetParallelScale(parallelScale*margin_factor)

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
    def frame_rate(self):
        rtis = self.GetLastRenderTimeInSeconds()
        fps = 1.0 / rtis
        return fps

    def fxaa_on(self):
        self.SetUseFXAA(True)

    def fxaa_off(self):
        self.SetUseFXAA(False)


class Renderer(Scene):
    """Your scene class.

    This is an important object that is responsible for preparing objects
    e.g. actors and volumes for rendering. This is a more pythonic version
    of ``vtkRenderer`` proving simple methods for adding and removing actors
    but also it provides access to all the functionality
    available in ``vtkRenderer`` if necessary.

    .. deprecated:: 0.2.0
          `Renderer()` will be removed in Fury v0.6.0, it is replaced by the
          class `Scene()`
    """

    @deprecate_with_version("Renderer() deprecated, Please use Scene()"
                            "instead", since='0.2.0', until='0.6.0')
    def __init__(self, _parent=None):
        """Init old class with a warning."""
        pass


@deprecate_with_version("'fury.window.renderer' function deprecated, use "
                        "'fury.window.Scene' instead",
                        since='0.2.0', until='0.6.0')
def renderer(background=None):
    """Create a Scene.

    .. deprecated:: 0.2.0
          `renderer` will be removed in Fury 0.6.0, it is replaced by the
          class `Scene()`

    Parameters
    ----------
    background : tuple
        Initial background color of scene

    Returns
    -------
    v : Scene instance
        scene object

    Examples
    --------
    >>> from fury import window, actor
    >>> import numpy as np
    >>> r = window.renderer()
    >>> lines=[np.random.rand(10,3)]
    >>> c=actor.line(lines, window.colors.red)
    >>> r.add(c)
    >>> #window.show(r)

    """
    scene = Scene()
    if background is not None:
        scene.SetBackground(background)

    return scene


@deprecate_with_version("'fury.window.ren' function deprecated, use "
                        "'fury.window.Scene' instead",
                        since='0.2.0', until='0.6.0')
def ren(background=None):
    """Create a Scene.

    .. deprecated:: 0.2.0
          `ren` will be removed in Fury 0.6.0, it is replaced by
          `Scene()`
    """
    return renderer(background=background)


@deprecate_with_version("'fury.window.add' function deprecated, use "
                        "'fury.window.Scene().add' instead",
                        since='0.2.0', until='0.6.0')
def add(scene, a):
    """Add a specific actor to the scene.

    .. deprecated:: 0.2.0
          `ren` will be removed in Fury 0.6.0, it is replaced by
          `Scene().add`
    """
    scene.add(a)


@deprecate_with_version("'fury.window.rm' function deprecated, use "
                        "'fury.window.Scene().rm' instead",
                        since='0.2.0', until='0.6.0')
def rm(scene, a):
    """Remove a specific actor from the scene.

    .. deprecated:: 0.2.0
          `ren` will be removed in Fury 0.6.0, it is replaced by
          `Scene().rm`
    """
    scene.rm(a)


@deprecate_with_version("'fury.window.clear' function deprecated, use "
                        "'fury.window.Scene().clear' instead",
                        since='0.2.0', until='0.6.0')
def clear(scene):
    """Remove all actors from the scene.

    .. deprecated:: 0.2.0
          `ren` will be removed in Fury 0.6.0, it is replaced by
          `Scene().clear`
    """
    scene.clear()


@deprecate_with_version("'fury.window.rm_all()' function deprecated, use "
                        "'fury.window.Scene().clear' instead",
                        since='0.2.0', until='0.6.0')
def rm_all(scene):
    """Remove all actors from the scene.

    .. deprecated:: 0.2.0
          `ren` will be removed in Fury 0.6.0, it is replaced by
          `Scene().rm_all`
    """
    scene.rm_all()


class ShowManager(object):
    """Class interface between the scene, the window and the interactor."""

    def __init__(self, scene=None, title='FURY', size=(300, 300),
                 png_magnify=1, reset_camera=True, order_transparent=False,
                 interactor_style='custom', stereo='off',
                 multi_samples=8, max_peels=4, occlusion_ratio=0.0):
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

        Methods
        -------
        initialize()
        render()
        start()
        add_window_callback()

        Examples
        --------
        >>> from fury import actor, window
        >>> scene = window.Scene()
        >>> scene.add(actor.axes())
        >>> showm = window.ShowManager(scene)
        >>> # showm.initialize()
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

        if self.reset_camera:
            self.scene.ResetCamera()

        self.window = vtk.vtkRenderWindow()

        if self.stereo.lower() != 'off':
            enable_stereo(self.window, self.stereo)

        self.window.AddRenderer(scene)

        self.window.SetSize(size[0], size[1])

        if self.order_transparent:
            occlusion_ratio = occlusion_ratio or 0.1
            antialiasing(self.scene, self.window,
                         multi_samples=0, max_peels=max_peels,
                         occlusion_ratio=occlusion_ratio)

        if self.interactor_style == 'image':
            self.style = vtk.vtkInteractorStyleImage()
        elif self.interactor_style == 'trackball':
            self.style = vtk.vtkInteractorStyleTrackballCamera()
        elif self.interactor_style == 'custom':
            self.style = CustomInteractorStyle()
        else:
            self.style = interactor_style

        self.iren = vtk.vtkRenderWindowInteractor()
        self.style.SetCurrentRenderer(self.scene)
        # Hack: below, we explicitly call the Python version of SetInteractor.
        self.style.SetInteractor(self.iren)
        self.iren.SetInteractorStyle(self.style)
        self.iren.SetRenderWindow(self.window)

    def initialize(self):
        """Initialize interaction."""
        self.iren.Initialize()

    def render(self):
        """Render only once."""
        self.window.Render()

    def start(self):
        """Start interaction."""
        try:
            self.render()
            if self.title.upper() == "FURY":
                self.window.SetWindowName(self.title + " " + fury_version)
            else:
                self.window.SetWindowName(self.title)
            self.iren.Start()
        except AttributeError:
            self.__init__(self.scene, self.title, size=self.size,
                          png_magnify=self.png_magnify,
                          reset_camera=self.reset_camera,
                          order_transparent=self.order_transparent,
                          interactor_style=self.interactor_style)
            self.initialize()
            self.render()
            if self.title.upper() == "FURY":
                self.window.SetWindowName(self.title + " " + fury_version)
            else:
                self.window.SetWindowName(self.title)
            self.iren.Start()

        self.window.RemoveRenderer(self.scene)
        self.scene.SetRenderWindow(None)
        self.window.Finalize()
        del self.iren
        del self.window

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
            filename = "recorded_events.log"
            recorder = vtk.vtkInteractorEventRecorder()
            recorder.SetInteractor(self.iren)
            recorder.SetFileName(filename)

            def _stop_recording_and_close(_obj, _evt):
                if recorder:
                    recorder.Stop()
                self.iren.TerminateApp()

            self.iren.AddObserver("ExitEvent", _stop_recording_and_close)

            recorder.EnabledOn()
            recorder.Record()

            self.initialize()
            self.render()
            self.iren.Start()
            # Deleting this object is the unique way
            # to close the file.
            recorder = None
            # Retrieved recorded events.
            with open(filename, 'r') as f:
                events = f.read()
        return events

    def record_events_to_file(self, filename="record.log"):
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
        if filename.endswith(".gz"):
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
        recorder = vtk.vtkInteractorEventRecorder()
        recorder.SetInteractor(self.iren)

        recorder.SetInputString(events)
        recorder.ReadFromInputStringOn()

        self.initialize()
        self.render()
        recorder.Play()

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
        if filename.endswith(".gz"):
            with gzip.open(filename, 'r') as f:
                events = f.read()
        else:
            with open(filename) as f:
                events = f.read()

        self.play_events(events)

    def add_window_callback(self, win_callback,
                            event=vtk.vtkCommand.ModifiedEvent):
        """Add window callbacks."""
        self.window.AddObserver(event, win_callback)
        self.window.Render()

    def add_timer_callback(self, repeat, duration, timer_callback):
        self.iren.AddObserver("TimerEvent", timer_callback)

        if repeat:
            timer_id = self.iren.CreateRepeatingTimer(duration)
        else:
            timer_id = self.iren.CreateOneShotTimer(duration)
        self.timers.append(timer_id)

    def destroy_timer(self, timer_id):
        self.iren.DestroyTimer(timer_id)
        del self.timers[self.timers.index(timer_id)]

    def destroy_timers(self):
        for timer_id in self.timers:
            self.destroy_timer(timer_id)

    def exit(self):
        """Close window and terminate interactor."""
        if is_osx and self.timers:
            # OSX seems to not destroy correctly timers
            # segfault 11 appears sometimes if we do not do it manually.
            self.destroy_timers()
        self.iren.GetRenderWindow().Finalize()
        self.iren.TerminateApp()
        self.timers.clear()


def show(scene, title='FURY', size=(300, 300), png_magnify=1,
         reset_camera=True, order_transparent=False, stereo='off',
         multi_samples=8, max_peels=4, occlusion_ratio=0.0):
    """Show window with current scene.

    Parameters
    ------------
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
    ----------
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

    See also
    ---------
    fury.window.record
    fury.window.snapshot

    """
    show_manager = ShowManager(scene, title, size, png_magnify, reset_camera,
                               order_transparent, stereo=stereo,
                               multi_samples=multi_samples,
                               max_peels=max_peels,
                               occlusion_ratio=occlusion_ratio)
    show_manager.initialize()
    show_manager.render()
    show_manager.start()


def record(scene=None, cam_pos=None, cam_focal=None, cam_view=None,
           out_path=None, path_numbering=False, n_frames=1, az_ang=10,
           magnification=1, size=(300, 300), reset_camera=True,
           screen_clip=False, stereo='off', verbose=False):
    """Record a video of your scene.

    Records a video as a series of ``.png`` files of your scene by rotating the
    azimuth angle az_angle in every frame.

    Parameters
    -----------
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
        How much to magnify the saved frame. Default is 1.
    size : (int, int)
        ``(width, height)`` of the window. Default is (300, 300).
    screen_clip: bool
        Clip the the png based on screen resolution. Default is False.
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
    ---------
    >>> from fury import window, actor
    >>> scene = window.Scene()
    >>> a = actor.axes()
    >>> scene.add(a)
    >>> # uncomment below to record
    >>> # window.record(scene)
    >>> # check for new images in current directory

    """
    if scene is None:
        scene = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.SetBorders(screen_clip)
    renWin.AddRenderer(scene)
    renWin.SetSize(size[0], size[1])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # scene.GetActiveCamera().Azimuth(180)

    if reset_camera:
        scene.ResetCamera()

    if stereo.lower() != 'off':
        enable_stereo(renWin, stereo)

    renderLarge = vtk.vtkRenderLargeImage()
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
        renderLarge = vtk.vtkRenderLargeImage()
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

        arr = numpy_support.vtk_to_numpy(renderLarge.GetOutput().GetPointData()
                                         .GetScalars())
        w, h, _ = renderLarge.GetOutput().GetDimensions()
        components = renderLarge.GetOutput().GetNumberOfScalarComponents()
        arr = np.flipud(arr.reshape((h, w, components)))
        save_image(arr, filename)

        ang = +az_ang


def antialiasing(scene, win, multi_samples=8, max_peels=4,
                 occlusion_ratio=0.0):
    """Enable anti-aliasing and ordered transparency.

    Parameters
    ----------
    scene : Scene
    win : Window
        Provided by Showmanager.window attribute.
    multi_samples : int
        Number of samples for anti-aliazing (Default 8).
        For no anti-aliasing use 0.
    max_peels : int
        Maximum number of peels for depth peeling (Default 4).
    occlusion_ratio : float
        Occlusion ration for depth peeling (Default 0 - exact image).
    """
    # Use a render window with alpha bits
    # as default is 0 (false))
    win.SetAlphaBitPlanes(True)

    # Force to not pick a framebuffer with a multisample buffer
    # (default is 8)
    win.SetMultiSamples(multi_samples)

    # Choose to use depth peeling (if supported)
    # (default is 0 (false)):
    scene.UseDepthPeelingOn()

    # Set depth peeling parameters
    # Set the maximum number of rendering passes (default is 4)
    scene.SetMaximumNumberOfPeels(max_peels)

    # Set the occlusion ratio (initial value is 0.0, exact image):
    scene.SetOcclusionRatio(occlusion_ratio)


def snapshot(scene, fname=None, size=(300, 300), offscreen=True,
             order_transparent=False, stereo='off',
             multi_samples=8, max_peels=4,
             occlusion_ratio=0.0):
    """Save a snapshot of the scene in a file or in memory.

    Parameters
    -----------
    scene : Scene() or vtkRenderer
        Scene instance
    fname : str or None
        Save PNG file. If None return only an array without saving PNG.
    size : (int, int)
        ``(width, height)`` of the window. Default is (300, 300).
    offscreen : bool
        Default True. Go stealthmode no window should appear.
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

    Returns
    -------
    arr : ndarray
        Color array of size (width, height, 3) where the last dimension
        holds the RGB values.

    """
    width, height = size

    render_window = vtk.vtkRenderWindow()
    if offscreen:
        render_window.SetOffScreenRendering(1)
    if stereo.lower() != 'off':
        enable_stereo(render_window, stereo)
    render_window.AddRenderer(scene)
    render_window.SetSize(width, height)

    if order_transparent:
        antialiasing(scene, render_window, multi_samples, max_peels,
                     occlusion_ratio)

    render_window.Render()

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = numpy_support.vtk_to_numpy(vtk_array).reshape(w, h, components)

    if fname is None:
        return arr

    save_image(arr, fname)
    return arr


def analyze_scene(scene):

    class ReportScene(object):
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


@deprecate_with_version("'fury.window.analyze_renderer' function deprecated, "
                        "use 'fury.window.analyze_scene' instead",
                        since='0.2.0', until='0.6.0')
def analyze_renderer(scene):
    """Report number of actors on the scene.

    .. deprecated:: 0.2.0
        `analyze_renderer` will be removed in Fury 0.3.0, it is replaced by
        `analyze_scene()`
    """
    return analyze_scene(scene)


def analyze_snapshot(im, bg_color=colors.black, colors=None,
                     find_objects=True,
                     strel=None):
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
        This is an object with attibutes like ``colors_found`` that give
        information about what was found in the current snapshot array ``im``.

    """
    if isinstance(im, basestring):
        im = load_image(im)

    class ReportSnapshot(object):
        objects = None
        labels = None
        colors_found = False

        def __str__(self):
            msg = "Report:\n-------\n"
            msg += "objects: {}\n".format(self.objects)
            msg += "labels: \n{}\n".format(self.labels)
            msg += "colors_found: {}\n".format(self.colors_found)
            return msg

    report = ReportSnapshot()

    if colors is not None:
        if isinstance(colors, tuple):
            colors = [colors]
        flags = [False] * len(colors)
        for (i, col) in enumerate(colors):
            # find if the current color exist in the array
            flags[i] = np.any(np.any(np.all(im[..., :3] == col[:3], axis=-1)))

        report.colors_found = flags

    if find_objects is True:
        weights = [0.299, 0.587, 0.144]
        gray = np.dot(im[..., :3], weights)
        bg_color = im[0, 0]
        background = np.dot(bg_color, weights)

        if strel is None:
            strel = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])

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
        'horizontal': 9
    }

    # default to horizontal since it is easy to see if it is working
    if stereo_type not in stereo_type_dictionary:
        warn("Unknown stereo type provided. "
             "Setting stereo type to 'horizontal'.")
        stereo_type = 'horizontal'

    renwin.SetStereoType(stereo_type_dictionary[stereo_type])
