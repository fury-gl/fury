"""FURY window module.

This module provides functionality for creating and managing
rendering windows using PyGfx. It includes classes and functions
for handling scenes, cameras, controllers, and rendering
multiple screens.
"""

import asyncio
from dataclasses import dataclass
from functools import reduce
import os
from typing import List

from PIL.Image import fromarray as image_from_array
import numpy as np

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
    Renderer,
    Scene as GfxScene,  # type: ignore
    ScreenCoordsCamera,
    Viewport,
    get_app,
    run,
)
from fury.ui import UI


class Scene(GfxScene):
    """Scene class to hold the actors in the scene.

    Data Structure to arrange the logical and spatial representation of the
    actors in the graphical scene. It is a subclass of PyGfx Scene class.
    It holds the background color and skybox texture. It also holds the lights in the
    scene. The background color and skybox texture can be set using the background
    property. The lights can be set using the lights property. The scene can be cleared
    using the clear method. The scene can be rendered using the render method.

    Parameters
    ----------
    background : tuple, optional
        The background color of the scene. It is a tuple of 4 floats (R, G, B, A).
    skybox : Texture, optional
        The skybox texture of the scene. It is a PyGfx Texture object.
    lights : list of Light, optional
        The lights in the scene. It is a list of PyGfx Light objects.
        If None, a default AmbientLight is added.
    """

    def __init__(
        self,
        *,
        background=(0, 0, 0, 1),
        skybox=None,
        lights=None,
    ):
        """Arrange the logical and spatial representation of actors.

        This class acts as a scene graph container, managing actors, background,
        and lighting for rendering.

        Parameters
        ----------
        background : tuple, optional
            Uniform color (R, G, B, A) for the scene background.
            Defaults to (0, 0, 0, 1).
        skybox : Texture, optional
            A PyGfx Texture object representing a cubemap for the background.
            If provided, overrides the `background` color. Defaults to None.
        lights : list of Light, optional
            A list of PyGfx Light objects to illuminate the scene. If None,
            a default AmbientLight is added. Defaults to None."""
        super().__init__()

        self.main_scene = GfxScene()

        self.ui_scene = GfxScene()
        self.ui_camera = ScreenCoordsCamera()
        self.ui_scene.add(self.ui_camera)
        self.add(self.ui_scene)

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
        """Create a skybox background actor from a cubemap texture.

        Parameters
        ----------
        cube_map : Texture
            A PyGfx Texture object (cubemap).

        Returns
        -------
        Background
            A PyGfx Background object configured with the skybox material."""
        return Background(
            geometry=None, material=BackgroundSkyboxMaterial(map=cube_map)
        )

    @property
    def background(self):
        """Get the background color of the scene.

        Returns
        -------
        tuple
            The current background color as an (R, G, B, A) tuple."""
        return self._bg_color

    @background.setter
    def background(self, value):
        """Set the background color of the scene.

        This replaces the current background actor (color or skybox)
        with a new uniform color background.

        Parameters
        ----------
        value : tuple
            The desired background color as an (R, G, B, A) tuple."""
        self.remove(self._bg_actor)
        self._bg_color = value
        self._bg_actor = Background.from_color(value)
        self.add(self._bg_actor)

    def set_skybox(self, cube_map):
        """Set a skybox as the scene background using a cubemap texture.

        This replaces the current background actor (color or skybox)
        with a new skybox background.

        Parameters
        ----------
        cube_map : Texture
            A PyGfx Texture object (cubemap) for the skybox."""
        self.remove(self._bg_actor)
        self._bg_actor = self._skybox(cube_map)
        self.add(self._bg_actor)

    def clear(self):
        """Remove all actors from the scene, keeping background and lights."""
        super().clear()
        self.add(self._bg_actor)
        self.add(*self.lights)

    def add(self, *objects):
        for object in objects:
            if isinstance(object, UI):
                object.add_to_scene(self.ui_scene)
            else:
                self.main_scene.add(object)


@dataclass
class Screen:
    """Define an independent viewport within the window.

    Holds a scene graph, camera, and controller for rendering actors
    within a specific rectangular area of the window."""

    viewport: Viewport
    scene: Scene
    camera: Camera
    controller: Controller

    @property
    def size(self):
        """Get the size of the screen viewport.

        Returns
        -------
        tuple
            The width and height (w, h) of the viewport in pixels."""
        return self.viewport.rect[2:]

    @property
    def position(self):
        """Get the position of the screen viewport within the window.

        Returns
        -------
        tuple
            The x and y coordinates (x, y) of the viewport's top-left corner."""
        return self.viewport.rect[:2]

    @property
    def bounding_box(self):
        """Get the bounding box of the screen viewport within the window.

        Returns
        -------
        tuple
            The position and size (x, y, w, h) of the viewport."""
        return self.viewport.rect

    @bounding_box.setter
    def bounding_box(self, value):
        """Set the bounding box of the screen viewport within the window.

        Parameters
        ----------
        value : tuple
            The desired position and size (x, y, w, h) for the viewport."""
        self.viewport.rect = value


def create_screen(
    renderer, *, rect=None, scene=None, camera=None, controller=None, camera_light=True
):
    """Compose a Screen object with viewport, scene, camera, and controller.

    Parameters
    ----------
    renderer : Renderer
        The PyGfx Renderer object associated with the window.
    rect : tuple, optional
        The bounding box (x, y, w, h) for the screen's viewport. If None,
        the viewport covers the entire renderer area initially. Defaults to None.
    scene : Scene, optional
        The scene graph to be rendered in this screen. If None, a new empty
        Scene is created. Defaults to None.
    camera : Camera, optional
        The PyGfx camera used to view the scene. If None, a PerspectiveCamera
        is created. Defaults to None.
    controller : Controller, optional
        The PyGfx controller for camera interaction. If None, an OrbitController
        is created and associated with the camera and viewport. Defaults to None.
    camera_light : bool, optional
        If True, attach a DirectionalLight to the camera. Defaults to True.

    Returns
    -------
    Screen
        A configured Screen object ready for rendering."""
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
    """Update the camera's view to encompass the target object or scene.

    If the target is a non-empty scene or another object, the camera adjusts
    to show it. If the target is an empty scene, the camera's aspect ratio
    is updated based on the provided size.

    Parameters
    ----------
    camera : Camera
        The PyGfx camera object to update.
    size : tuple
        The size (width, height) of the viewport, used if the target is empty.
    target : Object or Scene
        The PyGfx object or scene the camera should focus on."""
    if (isinstance(target, Scene) and len(target.children) > 3) or not isinstance(
        target, Scene
    ):
        camera.show_object(target)
    else:
        camera.width = size[0]
        camera.height = size[1]


def update_viewports(screens, screen_bbs):
    """Update the bounding boxes and cameras of multiple screens.

    Parameters
    ----------
    screens : list of Screen
        The list of Screen objects to update.
    screen_bbs : list of tuple
        A list of bounding boxes (x, y, w, h), one for each screen in `screens`."""
    for screen, screen_bb in zip(screens, screen_bbs, strict=False):
        screen.bounding_box = screen_bb
        update_camera(screen.camera, screen.size, screen.scene)


def render_screens(renderer, screens: List[Screen]):
    """Render multiple screens within a single renderer update cycle.

    Parameters
    ----------
    renderer : Renderer
        The PyGfx Renderer object to draw into.
    screens : list of Screen
        The list of Screen objects to render."""
    for screen in screens:
        scene_root = screen.scene
        screen.viewport.render(scene_root.main_scene, screen.camera, flush=False)
        screen.viewport.render(scene_root.ui_scene, scene_root.ui_camera, flush=False)

    renderer.flush()


def calculate_screen_sizes(screens, size):
    """Calculate screen bounding boxes based on a layout configuration.

    The `screens` list defines vertical sections, and each element within
    specifies the number of horizontal sections in that vertical column.

    Parameters
    ----------
    screens : list of int or None
        Layout configuration. Each integer represents a vertical column
        and specifies the number of horizontal rows within it. If None or
        empty, assumes a single screen covering the full size.
    size : tuple
        The total size (width, height) of the window or area to divide.

    Returns
    -------
    list of tuple
        A list of calculated bounding boxes (x, y, w, h) for each screen."""
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
    """Show manager for the rendering window.

    It manages the rendering of the scene(s) in the window. It also handles the
    events from the window and the controller.

    Parameters
    ----------
    renderer : Renderer
        The PyGfx Renderer object associated with the window.
    scene : Scene
        The scene graph to be rendered in the window.
    camera : Camera
        The PyGfx camera used to view the scene.
    controller : Controller
        The PyGfx controller for camera interaction.
    title : str
        The title of the window.
    size : tuple
        The size (width, height) of the window in pixels.
    blend_mode : str
        The blending mode used by the renderer.
    window_type : str
        The type of window canvas to create ('default', 'qt', 'jupyter', 'offscreen').
    pixel_ratio : float
        The ratio between render buffer and display buffer pixels.
    camera_light : bool
        Whether to attach a DirectionalLight to the camera.
    screen_config : list
        Defines the screen layout. Can be a list of integers (vertical/horizontal
        sections) or a list of explicit bounding box tuples (x, y, w, h).
    enable_events : bool
        Whether to enable mouse and keyboard interactions initially.
    qt_app : QApplication
        An existing QtWidgets QApplication instance (if `window_type` is 'qt').
    qt_parent : QWidget
        An existing QWidget to embed the QtCanvas within (if `window_type` is 'qt').
    """

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
        """Manage the rendering window, scenes, and interactions.

        Handles window creation, screen layout, rendering loop, and event handling.

        Parameters
        ----------
        renderer : Renderer, optional
            A PyGfx Renderer object. If None, a new one is created based on the
            `window_type`.
        scene : Scene or list of Scene, optional
            A single Scene object to be used for all screens, or a list of Scene
            objects, one for each screen defined by `screen_config`. If None,
            new Scene objects are created.
        camera : Camera or list of Camera, optional
            A single Camera object for all screens, or a list of Camera objects,
            one for each screen. If None, new PerspectiveCamera objects are
            created.
        controller : Controller or list of Controller, optional
            A single Controller for all screens, or a list of Controller objects,
            one for each screen. If None, new OrbitController objects are
            created.
        title : str, optional
            The title displayed in the window's title bar. Defaults to "FURY 2.0".
        size : tuple, optional
            The initial size (width, height) of the window in pixels.
        blend_mode : str, optional
            The blending mode used by the renderer. Accepted values include
            'default', 'additive', 'opaque', 'ordered1', 'ordered2', 'weighted',
            'weighted_depth', 'weighted_plus'.
        window_type : str, optional
            The type of window canvas to create. Accepted values are 'default'
            (or 'glfw'), 'qt', 'jupyter', 'offscreen'.
        pixel_ratio : float, optional
            The ratio between render buffer and display buffer pixels. Affects
            anti-aliasing and performance.
        camera_light : bool or list of bool, optional
            Whether to attach a DirectionalLight to the camera(s). Can be a
            single value for all screens or a list.
        screen_config : list, optional
            Defines the screen layout. Can be a list of integers (vertical/horizontal
            sections) or a list of explicit bounding box tuples (x, y, w, h).
            If None, assumes a single screen covering the window. Defaults to None.
        enable_events : bool, optional
            Whether to enable mouse and keyboard interactions initially.
        qt_app : QApplication, optional
            An existing QtWidgets QApplication instance (required if `window_type`
            is 'qt' and no global app exists).
        qt_parent : QWidget, optional
            An existing QWidget to embed the QtCanvas within (if `window_type`
            is 'qt')."""
        self._size = size
        self._title = title
        self._is_qt = False
        self._qt_app = qt_app
        self._qt_parent = qt_parent
        self._window_type = self._setup_window(window_type)

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer
        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.blend_mode = blend_mode
        self.renderer.add_event_handler(self._resize, "resize")
        self.renderer.add_event_handler(
            self._set_key_long_press_event, "key_down", "key_up"
        )

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
        self._key_long_press = None

    def _screen_setup(self, scene, camera, controller, camera_light):
        """Prepare scene, camera, controller, and light lists for screen creation.

        Ensures that lists match the total number of screens required.

        Parameters
        ----------
        scene : Scene or list of Scene or None
            Input scene configuration.
        camera : Camera or list of Camera or None
            Input camera configuration.
        controller : Controller or list of Controller or None
            Input controller configuration.
        camera_light : bool or list of bool
            Input camera light configuration."""
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
        """Initialize the appropriate canvas window based on the type.

        Parameters
        ----------
        window_type : str
            The requested window type ('default', 'glfw', 'qt', 'jupyter',
            'offscreen').

        Returns
        -------
        str
            The validated window type string.

        Raises
        ------
        ValueError
            If an invalid `window_type` is provided."""
        window_type = window_type.lower()

        if window_type not in ["default", "glfw", "qt", "jupyter", "offscreen"]:
            raise ValueError(
                f"Invalid window_type: {window_type}. "
                "Valid values are default, glfw, qt, jupyter, offscreen"
            )

        if window_type == "default" or window_type == "glfw":
            self.window = Canvas(size=self._size, title=self._title)
        elif window_type == "qt":
            if self._qt_app is None:
                self._qt_app = get_app()
            self.window = QtCanvas(
                size=self._size, title=self._title, parent=self._qt_parent
            )
            self._is_qt = True
        elif window_type == "jupyter":
            self.window = JupyterCanvas(size=self._size, title=self._title)
        else:
            self.window = OffscreenCanvas(size=self._size, title=self._title)

        return window_type

    def _calculate_total_screens(self):
        """Determine the total number of screens based on `screen_config`."""
        if self._screen_config is None or not self._screen_config:
            self._total_screens = 1
        elif isinstance(self._screen_config[0], int):
            self._total_screens = reduce(lambda a, b: a + b, self._screen_config)
        else:
            self._total_screens = len(self._screen_config)

    def _create_screens(self):
        """Create all Screen objects based on the prepared configurations.

        Returns
        -------
        list of Screen
            The list of created Screen objects."""
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
        """Handle window resize events by updating viewports and re-rendering.

        Parameters
        ----------
        _event : Event
            The PyGfx resize event object (unused in current implementation)."""
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )
        self.render()

    async def _handle_key_long_press(self, event):
        """Handle long press events for key inputs.

        Parameters
        ----------
        event : KeyEvent
            The PyGfx key event object."""

        if self._key_long_press is not None:
            await asyncio.sleep(0.05)
            self.renderer.dispatch_event(event)

    def _set_key_long_press_event(self, event):
        """Handle long press events for key inputs.

        Parameters
        ----------
        event : KeyEvent
            The PyGfx key event object."""

        if event.type == "key_down":
            self._key_long_press = asyncio.create_task(
                self._handle_key_long_press(event)
            )
        else:
            self._key_long_press.cancel()
            self._key_long_press = None

    @property
    def app(self):
        """Get the associated QApplication instance, if any.

        Returns
        -------
        QApplication or None
            The QApplication instance if the window type is 'qt', otherwise None."""
        return self._qt_app

    @property
    def title(self):
        """Get the current window title.

        Returns
        -------
        str
            The text displayed in the window's title bar."""
        return self._title

    @title.setter
    def title(self, value):
        """Set the window title.

        Parameters
        ----------
        value : str
            The desired text for the window's title bar."""
        self._title = value
        self.window.set_title(self._title)

    @property
    def pixel_ratio(self):
        """Get the current pixel ratio of the renderer.

        Returns
        -------
        float
            The ratio between render buffer and display buffer pixels."""
        return self.renderer.pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        """Set the pixel ratio of the renderer.

        Parameters
        ----------
        value : float
            The desired pixel ratio."""
        self.renderer.pixel_ratio = value

    @property
    def size(self):
        """Get the current size of the window.

        Returns
        -------
        tuple
            The current (width, height) of the window in logical pixels."""
        return self._size

    def set_enable_events(self, value):
        """Enable or disable mouse and keyboard interactions for all screens.

        Parameters
        ----------
        value : bool
            Set to True to enable events, False to disable them."""
        self.enable_events = value
        if value:
            self.renderer.enable_events()
        else:
            self.renderer.disable_events()
        for s in self.screens:
            s.controller.enabled = value

    def snapshot(self, fname):
        """Save a snapshot of the current rendered content to a file.

        The window must have been rendered at least once before calling this.

        Parameters
        ----------
        fname : str
            The file path (including extension, e.g., 'image.png') where the
            snapshot will be saved.

        Returns
        -------
        ndarray
            A NumPy array representing the captured image data (RGBA)."""
        arr = np.asarray(self.renderer.snapshot())
        img = image_from_array(arr)
        img.save(fname)
        return arr

    def _draw_function(self):
        render_screens(self.renderer, self.screens)
        self.window.request_draw()

    def render(self):
        """Request a redraw of all screens in the window."""
        if self._is_qt and self._qt_parent is not None:
            self._qt_parent.show()
        self.window.request_draw(self._draw_function)

    def start(self):
        """Start the rendering event loop and display the window.

        This call blocks until the window is closed, unless running in an
        offscreen or specific environment (like FURY_OFFSCREEN)."""
        self.render()
        if "FURY_OFFSCREEN" in os.environ and os.environ["FURY_OFFSCREEN"].lower() in [
            "true",
            "1",
        ]:
            self.window.draw_frame()
            self.snapshot(f"{self._title}.png")
            return

        if self._is_qt:
            self._qt_app.exec()
        else:
            run()

    def close(self):
        """Close the rendering window and terminate the application if necessary."""
        self.window.close()


def snapshot(
    *,
    scene=None,
    screen_config=None,
    fname="output.png",
    actors=None,
    return_array=False,
):
    """Take a snapshot using an offscreen window.

    Creates a temporary offscreen ShowManager, renders the scene(s),
    saves the image, and optionally returns the image data.

    Parameters
    ----------
    scene : Scene or list of Scene, optional
        The scene(s) to render. If `actors` is provided, this is ignored.
        Defaults to None.
    screen_config : list, optional
        Screen layout configuration (see ShowManager). Defaults to None (single screen).
    fname : str, optional
        The file path to save the snapshot image. Defaults to "output.png".
    actors : Object or list of Object, optional
        Convenience parameter. If provided, a new Scene is created containing
        these actors, and the `scene` parameter is ignored. Defaults to None.
    return_array : bool, optional
        If True, the function returns the image data as a NumPy array in addition
        to saving the file. Defaults to False.

    Returns
    -------
    ndarray or None
        If `return_array` is True, returns the RGBA image data as a NumPy array.
        Otherwise, returns None."""
    if actors is not None:
        scene = Scene()
        scene.add(*actors)

    show_m = ShowManager(
        scene=scene, screen_config=screen_config, window_type="offscreen"
    )
    show_m.render()
    show_m.window.draw_frame()
    arr = show_m.snapshot(fname)

    if return_array:
        return arr


def show(actors, *, window_type="default"):
    """Display one or more actors in a new window quickly.

    A convenience function to quickly visualize actors without manually
    setting up a Scene or ShowManager.

    Parameters
    ----------
    actors : Object or list of Object
        The PyGfx actor(s) to display.
    window_type : str, optional
        The type of window canvas to create ('default', 'glfw', 'qt',
        'jupyter', 'offscreen'). Defaults to 'default'."""
    scene = Scene()
    scene.add(*actors)
    show_m = ShowManager(scene=scene, window_type=window_type)
    show_m.start()
