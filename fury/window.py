"""FURY window module.

This module provides functionality for creating and managing
rendering windows using PyGfx. It includes classes and functions
for handling scenes, cameras, controllers, and rendering
multiple screens.
"""

import asyncio
from dataclasses import dataclass
from functools import reduce
import logging
import os

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
    EventType,
    Group as GfxGroup,  # type: ignore
    JupyterCanvas,
    OffscreenCanvas,
    PerspectiveCamera,
    PointerEvent,
    QtCanvas,
    Renderer,
    Scene as GfxScene,  # type: ignore
    ScreenCoordsCamera,
    Stats,
    TrackballController,
    Viewport,
    call_later,
    get_app,
    qcall_later,
    run,
)
from fury.ui import UI, UIContext


class Scene(GfxGroup):
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
        self.ui_elements = []

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
        self.main_scene.clear()
        self.main_scene.add(self._bg_actor)
        self.main_scene.add(*self.lights)

        self.ui_elements.clear()
        self.ui_scene.clear()
        self.ui_scene.add(self.ui_camera)

    def add(self, *objects):
        """Add actors or UI elements to the scene.

        Parameters
        ----------
        *objects : list of Mesh or UI
            A list objects to be added to the scene.
        """
        for obj in objects:
            if isinstance(obj, UI):
                self.ui_elements.append(obj)
                add_ui_to_scene(self.ui_scene, obj)
            elif isinstance(obj, GfxScene):  # type: ignore [misc]
                super().add(obj)
            else:
                self.main_scene.add(obj)

    def remove(self, *objects):
        """Remove actors or UI elements from the scene.

        Parameters
        ----------
        *objects : list of Mesh or UI
            A list of objects to be removed from the scene.
        """
        for obj in objects:
            if isinstance(obj, UI):
                if obj in self.ui_elements:
                    self.ui_elements.remove(obj)
                remove_ui_from_scene(self.ui_scene, obj)
            elif isinstance(obj, GfxScene):  # type: ignore [misc]
                super().remove(obj)
            else:
                self.main_scene.remove(obj)


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


def add_ui_to_scene(ui_scene, ui_obj):
    """Recursively traverse and add UI hierarchy to the UI scene.

    Parameters
    ----------
    ui_scene : GfxScene
        Scene dedicated to UI elements.
    ui_obj : UI
        UI element to add into scene.
    """
    if ui_obj.actors:
        ui_scene.add(*ui_obj.actors)

    for child in ui_obj._children:
        add_ui_to_scene(ui_scene, child)


def remove_ui_from_scene(ui_scene, ui_obj):
    """Recursively traverse and remove UI hierarchy from the UI scene.

    Parameters
    ----------
    ui_scene : GfxScene
        Scene dedicated to UI elements.
    ui_obj : UI
        UI element to be removed from the scene.
    """
    if ui_obj.actors:
        ui_scene.remove(*ui_obj.actors)

    for child in ui_obj._children:
        remove_ui_from_scene(ui_scene, child)


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
        controller = TrackballController(camera, register_events=vp)

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
    if isinstance(target, Scene):
        target = target.main_scene

    if (isinstance(target, GfxScene) and len(target.children) > 3) or (  # type: ignore [misc]
        not isinstance(target, GfxScene) and target is not None  # type: ignore [misc]
    ):
        camera.show_object(target)
    elif size is not None:
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


def render_screens(renderer, screens, stats=None):
    """Render multiple screens within a single renderer update cycle.

    Parameters
    ----------
    renderer : Renderer
        The PyGfx Renderer object to draw into.
    screens : list of Screen
        The list of Screen objects to render.
    stats : Stats, optional
        Stats helper to display FPS overlay."""
    if stats is not None:
        stats.start()

    for screen in screens:
        scene_root = screen.scene
        screen.viewport.render(scene_root.main_scene, screen.camera, flush=False)
        screen.viewport.render(scene_root.ui_scene, scene_root.ui_camera, flush=False)

    if stats is not None:
        stats.stop()
        stats.render(flush=False)

    renderer.flush()


def reposition_ui(screens):
    """Update the positions of all UI elements across multiple screens.

    Parameters
    ----------
    screens : list of Screen
        The list of Screen objects containing UI elements to reposition.
    """

    for screen in screens:
        scene_root = screen.scene
        for child in scene_root.ui_elements:
            child._update_actors_position()


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
    show_fps : bool
        Whether to display FPS statistics using an on-screen overlay.
    max_fps : int
        Maximum frames per second for the canvas.
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
        window_type="default",
        pixel_ratio=1,
        camera_light=True,
        screen_config=None,
        enable_events=True,
        qt_app=None,
        qt_parent=None,
        show_fps=False,
        max_fps=60,
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
            is 'qt').
        show_fps : bool, optional
            Whether to display FPS statistics in the renderer.
        max_fps : int, optional
            Maximum frames per second for the canvas.
        """
        self._size = size
        self._title = title
        self._is_qt = False
        self._qt_app = qt_app
        self._qt_parent = qt_parent
        self._is_initial_resize = None
        self._show_fps = show_fps
        self._max_fps = max_fps
        self._window_type = self._setup_window(window_type)
        self._is_dragging = False
        self._drag_target = None

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer
        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.add_event_handler(
            lambda event: self._resize(size=(event.width, event.height)),
            EventType.RESIZE,
        )
        self.renderer.add_event_handler(
            self._set_key_long_press_event, EventType.KEY_DOWN, EventType.KEY_UP
        )
        self.renderer.add_event_handler(
            self._register_drag,
            EventType.POINTER_DOWN,
            EventType.POINTER_UP,
            EventType.POINTER_MOVE,
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
        self._callbacks = {}

        self._stats = None
        self._stats_initialized = False

        self.enable_events = enable_events
        self._key_long_press = None
        self._resize(self._size)

    def _handle_drag(self, event):
        """Handle drag events for pointer interactions.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if self._drag_target is None:
            self._drag_target = event.target
        drag_event = PointerEvent(
            x=event.x, y=event.y, type=EventType.POINTER_DRAG, target=self._drag_target
        )
        self.renderer.dispatch_event(drag_event)

    def _register_drag(self, event):
        """Register drag events for pointer interactions.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """

        if event.type == EventType.POINTER_DOWN:
            self._is_dragging = True
            self._drag_target = event.target
        elif event.type == EventType.POINTER_UP:
            self._is_dragging = False
            self._drag_target = None
        elif event.type == EventType.POINTER_MOVE and self._is_dragging:
            self._handle_drag(event)

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
            self.window = Canvas(
                size=self._size, title=self._title, max_fps=self._max_fps
            )
        elif window_type == "qt":
            self.window = QtCanvas(
                size=self._size,
                title=self._title,
                parent=self._qt_parent,
                max_fps=self._max_fps,
            )
            self._is_qt = True
        elif window_type == "jupyter":
            self.window = JupyterCanvas(
                size=self._size, title=self._title, max_fps=self._max_fps
            )
        else:
            self.window = OffscreenCanvas(
                size=self._size, title=self._title, max_fps=self._max_fps
            )

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

    def _resize(self, size):
        """Handle window resize events by updating viewports and re-rendering.

        Parameters
        ----------
        size : tuple
            The size (width, height) of the window in pixels.
        """
        UIContext.canvas_size = size
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )
        reposition_ui(self.screens)
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

        if event.type == EventType.KEY_DOWN:
            self._key_long_press = asyncio.create_task(
                self._handle_key_long_press(event)
            )
        elif self._key_long_press is not None:
            self._key_long_press.cancel()
            self._key_long_press = None

    def _on_repeat_callback(self, func, time, name, *args):
        """Internal method to handle the timing and execution of callbacks.

        Parameters
        ----------
        func : callable
            The function to be called.
        time : float
            The time interval in seconds after which the function is called.
        name : str
            A unique name for the callback.
        *args : tuple
            Additional arguments to pass to the function.
        """
        args = (func, time, name, *args)

        if name not in self._callbacks:
            return

        if self._is_qt:
            qcall_later(time, self._on_repeat_callback, *args)
        else:
            call_later(time, self._on_repeat_callback, *args)
        func(*args[3:])

    def register_callback(self, func, time, repeat, name, *args):
        """Register a callback function to be called after a time interval.

        Parameters
        ----------
        func : callable
            The function to be called.
        time : float
            The time interval in seconds after which the function is called.
        repeat : bool
            If True, the function is called repeatedly every `time` seconds.
            If False, it is called only once.
        name : str
            A unique name for the callback.
        *args : tuple
            Additional arguments to pass to the function.
        """
        if repeat:
            if name in self._callbacks:
                logging.warning(
                    f"Callback with name '{name}' is already registered."
                    "Please use a different name."
                )
                return

            self._callbacks[name] = (func, time, repeat, args)

            args = (func, time, name, *args)
            if self._is_qt:
                qcall_later(time, self._on_repeat_callback, *args)
            else:
                call_later(time, self._on_repeat_callback, *args)
        else:
            if self._is_qt:
                qcall_later(time, func, *args)
            else:
                call_later(time, func, *args)

    def cancel_callback(self, name):
        """Cancel a registered callback by its name.

        Parameters
        ----------
        name : str
            The unique name of the callback to cancel.
        """
        if name in self._callbacks:
            del self._callbacks[name]

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

    @property
    def callbacks(self):
        """Get the registered callbacks.

        This only returns the callbacks that are set to repeat.

        Returns
        -------
        dict
            A dictionary of registered callbacks with their names as keys."""
        return self._callbacks

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

    def get_fps(self):
        """Get the current FPS from the stats overlay if available.

        Returns
        -------
        int or None
            The current FPS value, or None if stats are not initialized or
            FPS has not been computed yet."""
        if self._stats is not None:
            return getattr(self._stats, "_fps", None)
        return None

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
            A NumPy array representing the captured image data (RGBA).
        """
        arr = np.asarray(self.renderer.snapshot())
        img = image_from_array(arr)
        img.save(fname)
        return arr

    def _draw_function(self):
        """Draw all screens and request a window redraw."""
        if self._show_fps and not self._stats_initialized:
            self._stats = Stats(self.renderer)
            self._stats_initialized = True

        render_screens(self.renderer, self.screens, stats=self._stats)
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
            self._draw_function()
            self.snapshot(f"{self._title}.png")
            self.window.close()
            return

        if self._is_qt:
            if self._qt_app is None:
                self._qt_app = get_app()
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
    show_m.window.draw()
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
