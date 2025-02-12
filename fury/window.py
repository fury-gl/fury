from dataclasses import dataclass
from functools import reduce

from PIL.Image import fromarray as image_from_array
import numpy as np
from scipy import ndimage

from fury.io import load_image
from fury.lib import (
    AmbientLight,
    Background,
    BackgroundSkyboxMaterial,
    Camera,
    Canvas,
    Controller,
    DirectionalLight,
    JupyterCanvas,
    OffscreenCanvas,
    OrbitController,
    PerspectiveCamera,
    QtCanvas,
    QtWidgets,
    Renderer,
    Scene as GfxScene,  # type: ignore
    Viewport,
    run,
)


class Scene(GfxScene):
    def __init__(
        self,
        *,
        background=(0, 0, 0, 1),
        skybox=None,
        lights=None,
    ):
        """Data Structure to arrange the logical and spatial representation of the
        actors in the graphical scene.

        Parameters
        ----------
        background : tuple, optional
            Uniform color to show in the background of scene, by default (0, 0, 0, 1)
        skybox : Texture, optional
            PyGfx Texture object
        lights : list, optional
            list of Light objects to add to the scene. If not passed AmbientLight is
            added to the scene
        """
        super().__init__()

        self._bg_color = background
        self._bg_actor = None

        if skybox is not None:
            self._bg_actor = self._skybox(skybox)
        else:
            self._bg_actor = Background.from_color(background)

        self.add(self._bg_actor)

        self.lights = lights
        if self.lights is None:
            self.lights = []
            self.lights.append(AmbientLight())

        self.add(*self.lights)

    def _skybox(self, cube_map):
        """Create skybox background from cubemap.

        Parameters
        ----------
        cube_map : Texture
            PyGfx Texture object

        Returns
        -------
        Background
            PyGfx background object
        """
        return Background(
            geometry=None, material=BackgroundSkyboxMaterial(map=cube_map)
        )

    @property
    def background(self):
        """Get background Color of the scene.

        Returns
        -------
        tuple
            (R, G, B, A) tuple
        """
        return self._bg_color

    @background.setter
    def background(self, value):
        """Set background color of the scene.

        Parameters
        ----------
        value : tuple
            (R, G, B, A) tuple
        """
        self.remove(self._bg_actor)
        self._bg_color = value
        self._bg_actor = Background.from_color(value)
        self.add(self._bg_actor)

    def set_skybox(self, cube_map):
        """Set skybox from cubemap as background.

        Parameters
        ----------
        cube_map : Texture
            PyGfx Texture object
        """
        self.remove(self._bg_actor)
        self._bg_actor = self._skybox(cube_map)
        self.add(self._bg_actor)

    def clear(self):
        """Removes all the children from the scene graph."""
        super().clear()
        self.add(self._bg_actor)
        self.add(*self.lights)


@dataclass
class Screen:
    """Define an independent viewport to show in the window, it holds a scene graph to
    show actors in the defined space.
    """

    viewport: Viewport
    scene: Scene
    camera: Camera
    controller: Controller

    @property
    def size(self):
        """Size of the screen.

        Returns
        -------
        tuple
            (w, h)
        """
        return self.viewport.rect[2:]

    @property
    def position(self):
        """Position of the screen in the window.

        Returns
        -------
        tuple
            (x, y)
        """
        return self.viewport.rect[:2]

    @property
    def bounding_box(self):
        """Bounding box of the screen in the window.

        Returns
        -------
        tuple
            (x, y, w, h)
        """
        return self.viewport.rect

    @bounding_box.setter
    def bounding_box(self, value):
        """Set bounding box of the screen in the window.

        Parameters
        ----------
        value : tuple
            (x, y, w, h)
        """
        self.viewport.rect = value


def create_screen(
    renderer, *, rect=None, scene=None, camera=None, controller=None, camera_light=True
):
    """Compose a screen.

    Parameters
    ----------
    renderer : Renderer
        PyGfx Renderer object to hold the viewport of the screen
    rect : tuple, optional
        Bounding box of (x, y, w, h)
    scene : Scene, optional
        scene graph of the screen. If None, a new scene is created
        PyGfx camera of choice to visualize. If None, Perspective Camera is used
    controller : Controller, optional
        PyGfx Controller of choice to visualize. If None, Orbit Controller is used
    camera_light : bool, optional
        If True a directional light is attached on top of the camera

    Returns
    -------
    Screen
        screen object to render.
    """
    vp = Viewport(renderer, rect)
    if scene is None:
        scene = Scene()
    if camera is None:
        camera = PerspectiveCamera(50)
        if camera_light:
            light = DirectionalLight()
            camera.add(light)
            scene.add(camera)

    if controller is None:
        controller = OrbitController(camera, register_events=vp)

    screen = Screen(vp, scene, camera, controller)
    update_camera(camera, screen.size, scene)
    return screen


def update_camera(camera, size, target):
    """Updates the camera to face the target. The size will be used when the target is
    an empty scene.

    Parameters
    ----------
    camera : Camera
        PyGfx camera object to face the target
    size : tuple
        Size of the target object
    target : Object
        PyGfx Object to show on camera.
    """

    if (isinstance(target, Scene) and len(target.children) > 3) or not isinstance(
        target, Scene
    ):
        camera.show_object(target)
    else:
        camera.width = size[0]
        camera.height = size[1]


def update_viewports(screens, screen_bbs):
    """Update the screen viewports to given bounding boxes.

    Parameters
    ----------
    screens : list
        list of Screen objects
    screen_bbs : list
        list of tuple of bounding boxes
    """
    for screen, screen_bb in zip(screens, screen_bbs):
        screen.bounding_box = screen_bb
        update_camera(screen.camera, screen.size, screen.scene)


def render_screens(renderer, screens):
    """Render screens in renderer on update.

    Parameters
    ----------
    renderer : Renderer
        PyGfx Renderer to update
    screens : list
        list of Screen objects
    """
    for screen in screens:
        screen.viewport.render(screen.scene, screen.camera, flush=False)

    renderer.flush()


def calculate_screen_sizes(screens, size):
    """Calculate the screen sizes based on the configurations of the screens.

    Parameters
    ----------
    screens : list
        List of screen config. Each entry in the list indicates a vertical section. Each
        value in the list indicates horizontal sections in the respective vertical
        section.
    size : tuple
        Size of the window

    Returns
    -------
    list
        list of bounding box of each screen.
    """
    if screens is None or not screens:
        return [(0, 0, *size)]

    screen_bbs = []

    v_sections = len(screens)
    width = (1 / v_sections) * size[0]
    x = 0

    for h_section in screens:
        if h_section == 0:
            continue

        height = (1 / h_section) * size[1]
        y = 0

        for _ in range(h_section):
            screen_bbs.append((x, y, width, height))
            y += height

        x += width

    return screen_bbs


class ShowManager:
    def __init__(
        self,
        *,
        renderer=None,
        scene=None,
        camera=None,
        controller=None,
        title="FURY 2.0",
        size=(800, 800),
        blend_mode="default",
        window_type="default",
        pixel_ratio=1,
        camera_light=True,
        screen_config=None,
        enable_events=True,
        qt_app=None,
        qt_parent=None,
    ):
        """Show manager for the rendering window.

        Parameters
        ----------
        renderer : Renderer, optional
            PyGfx Renderer object. If None, a new Renderer object is created
        scene : Scene or list, optional
            If scene is passed same scene is applied to all the screens.
            Else each scene is provided to respective screen from the list.
            If None, a new scene object is created.
        camera : Camera or list, optional
            If camera is passed same camera is applied to all the screens.
            Else each camera is provided to respective screen from the list.
            If None, a new camera object is created.
        controller : Controller or list, optional
            If controller is passed same controller is applied to all the screens.
            Else each controller is provided to respective screen from the list.
            If None, a new controller object is created.
        title : str, optional
            Title of the window.
        size : tuple, optional
            Size of the window.
        blend_mode : str, optional
            Renderer blend mode. One of the following blend mode is accepted
            - additive or default : single-pass approach that adds fragments together.
            - opaque : single-pass approach that ignores transparency.
            - ordered1 : single-pass approach that blends fragments (using alpha
            blending). Can only produce correct results if fragments are drawn
            from back to front.
            - ordered2 : two-pass approach that first processes all opaque
            fragments and then blends transparent fragments (using alpha blending)
            with depth-write disabled. The visual results are usually better than
            ordered1, but still depend on the drawing order.
            - weighted : two-pass approach for order independent transparency based
            on alpha weights.
            - weighted_depth : two-pass approach for order independent transparency
            based on alpha weights and depth [1]. Note that the depth range
            affects the (quality of the) visual result.
            - weighted_plus : three-pass approach for order independent
            transparency, in which the front-most transparent layer is rendered
            correctly, while transparent layers behind it are blended using alpha
            weights.
        window_type : str, optional
            Type of the window. One of the following window type is accepted
            - glfw or default: select default GLFW canvas window.
            - qt: select Qt canvas window.
            - jupyter: select jupyter_rfb canvas widget.
            - offscreen: select offscreen canvas to not show any window for remote runs.
        pixel_ratio : float, optional
            The ratio between the number of pixels in the render buffer versus the
            number of pixels in the display buffer. If None, this will be 1 for high-res
            canvases and 2 otherwise. If greater than 1, SSAA (supersampling
            anti-aliasing) is applied while converting a render buffer to a display
            buffer. If smaller than 1, pixels from the render buffer are replicated
            while converting to a display buffer. This has positive performance
            implications.
        camera_light : bool, optional
            To attach a light on top of camera
        screen_config : list, optional
            List of all the vertical and horizontal section or list of all the bounding
            boxes of the screens. If None, single screen is assumed.
        enable_events : bool, optional
            Enable the events from mouse and keyboard on the visualization.
        qt_app : QApplication, optional
            QtWidgets QApplication object for QtCanvas.
        qt_parent : QWidget, optional
            QWidget object for putting the window in a QLayout.
        """
        self._size = size
        self._title = title
        self._is_qt = False
        self._qt_app = qt_app
        self._qt_parent = qt_parent
        self._window_type = window_type
        self._setup_window(window_type)

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer
        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.blend_mode = blend_mode
        self.renderer.add_event_handler(self._resize, "resize")

        self._total_screens = 0
        self._screen_config = screen_config
        self._calculate_total_screens()
        self._screen_setup(scene, camera, controller, camera_light)
        self.screens = self._create_screens()
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )

        self.enable_events = enable_events

    def _screen_setup(self, scene, camera, controller, camera_light):
        """Setup to create the screens.

        Parameters
        ----------
        scene : Scene
        camera : Camera
        controller : Controller
        camera_light : bool
            If True, attach a direction light with the camera
        """
        self._scene = scene
        if not isinstance(scene, list):
            self._scene = [scene] * self._total_screens

        self._camera = camera
        if not isinstance(camera, list):
            self._camera = [camera] * self._total_screens

        self._controller = controller
        if not isinstance(controller, list):
            self._controller = [controller] * self._total_screens

        self._camera_light = camera_light
        if not isinstance(camera_light, list):
            self._camera_light = [camera_light] * self._total_screens

    def _setup_window(self, window_type):
        """Initialize the canvas window.

        Parameters
        ----------
        window_type : str
            Type of the window. One of the following window type is accepted
            - glfw or default : select default GLFW canvas window.
            - qt : select Qt canvas window.
            - jupyter : select jupyter_rfb canvas widget.
            - offscreen : select offscreen canvas to not show any window for remote runs.
        """
        window_type = window_type.lower()

        if window_type not in ["default", "glfw", "qt", "jupyter", "offscreen"]:
            raise ValueError(
                "Invalid window_type: {}. "
                "Valid values are default, glfw, qt, jupyter, offscreen".format(
                    window_type
                )
            )

        if window_type == "default" or window_type == "glfw":
            self.window = Canvas(size=self._size, title=self._title)
        elif window_type == "qt":
            if self._qt_app is None:
                self._qt_app = QtWidgets.QApplication([])
            self.window = QtCanvas(
                size=self._size, title=self._title, parent=self._qt_parent
            )
            self._is_qt = True
        elif window_type == "jupyter":
            self.window = JupyterCanvas(size=self._size, title=self._title)
        else:
            self.window = OffscreenCanvas(size=self._size, title=self._title)

    def _calculate_total_screens(self):
        """Calculate the total screens from the screen configurations."""
        if self._screen_config is None or not self._screen_config:
            self._total_screens = 1
        elif isinstance(self._screen_config[0], int):
            self._total_screens = reduce(lambda a, b: a + b, self._screen_config)
        else:
            self._total_screens = len(self._screen_config)

    def _create_screens(self):
        """Create screens from screen setup.

        Returns
        -------
        list
            list of Screen objects
        """
        screens = []
        for i in range(self._total_screens):
            screens.append(
                create_screen(
                    self.renderer,
                    scene=self._scene[i],
                    camera=self._camera[i],
                    controller=self._controller[i],
                    camera_light=self._camera_light[i],
                )
            )
        return screens

    def _resize(self, _event):
        """Resize the inner screens based on the event on the window.

        Parameters
        ----------
        _event : Event
            PyGfx Event object for window.
        """
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )
        self.render()

    @property
    def app(self):
        """QtApplication

        Returns
        -------
        QtApplication or None
            If window type is qt returns QtApplication object else returns None.
        """
        return self._qt_app

    @property
    def title(self):
        """Title of the window.

        Returns
        -------
        str
        """
        return self._title

    @title.setter
    def title(self, value):
        """Set title of the window.

        Parameters
        ----------
        value : str
        """
        self._title = value
        self.window.set_title(self._title)

    @property
    def pixel_ratio(self):
        """Pixel ratio of the renderer.

        Returns
        -------
        float
        """
        return self.renderer.pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        """Set pixel ratio of the renderer.

        Parameters
        ----------
        value : float
        """
        self.renderer.pixel_ratio = value

    @property
    def size(self):
        """Size of the window.

        Returns
        -------
        tuple
            (w, h) of the window.
        """
        return self._size

    def set_enable_events(self, value):
        """Enable or disables the events on the rendering window.

        Parameters
        ----------
        value : bool
        """
        self.enable_events = value
        if value:
            self.renderer.enable_events()
        else:
            self.renderer.disable_events()
        for s in self.screens:
            s.controller.enabled = value

    def snapshot(self, fname):
        """Save a copy of the rasterized image of the scene(s).
        The window_type needs to be offscreen for snapshot to work.

        Parameters
        ----------
        fname : str
            file name or path to store the file

        Returns
        -------
        narray
            numpy array of the image.
        """
        if self._window_type != "offscreen":
            raise ValueError(
                "Invalid window_type: {}. "
                "Snapshot functionality is only allowed on offscreen".format(
                    self._window_type
                )
            )
        self.render()
        self.window.draw()
        arr = np.asarray(self.renderer.snapshot())
        img = image_from_array(arr)
        img.save(fname)
        return arr

    def render(self):
        """Rasterize the scene(s)."""
        self.window.request_draw(lambda: render_screens(self.renderer, self.screens))

    def start(self):
        """Start the visualization using show manager."""
        self.render()
        if self._is_qt:
            self._qt_app.exec()
        else:
            run()


def snapshot(
    *,
    scene=None,
    screen_config=None,
    fname="output.png",
    actors=None,
    return_array=False,
):
    """Save a snapshot of the rasterized image of the window.

    Parameters
    ----------
    scene : Scene or list, optional
        scene(s) graph to capture the image.
    screen_config : list, optional
        List of all the vertical and horizontal section or list of all the bounding
        boxes of the screens. If None, single screen is assumed.
    fname : str, optional
        Name or path of the output image file.
    actors : Object, optional
        PyGfx Objects to show on the scene. Works with single scene configuration.
    return_array : bool, optional
        If True, return the numpy array of the image.
    Returns
    -------
    narray
        numpy array of the image
    """
    if actors is not None:
        scene = Scene()
        scene.add(*actors)

    show_m = ShowManager(
        scene=scene, screen_config=screen_config, window_type="offscreen"
    )
    arr = show_m.snapshot(fname)

    if return_array:
        return arr


def show(actors, *, window_type="default"):
    """Display given actors in a fury window. A Quick way to visualize the actors.

    Parameters
    ----------
    actors : Object
        PyGfx Object
    window_type : str, optional
        Type of the window. One of the following window type is accepted
        - glfw or default : select default GLFW canvas window.
        - qt : select Qt canvas window.
        - jupyter : select jupyter_rfb canvas widget.
        - offscreen : select offscreen canvas to not show any window for remote runs.
    """
    scene = Scene()
    scene.add(*actors)
    show_m = ShowManager(scene=scene, window_type=window_type)
    show_m.start()
