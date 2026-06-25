"""
FURY window module.

This module provides functionality for creating and managing rendering
windows using PyGfx. It includes classes and functions for handling
scenes, cameras, controllers, and rendering multiple screens.
"""

from dataclasses import dataclass
from functools import reduce
import logging
import os
import sys
from typing import Iterable

from PIL.Image import fromarray as image_from_array
import numpy as np
from scipy import ndimage

from fury.actor import Group
from fury.actor.core import create_axes_helper
from fury.animation.animation import Animation, CameraAnimation
from fury.animation.timeline import Timeline
from fury.io import load_image
from fury.lib import (
    AmbientLight,
    Background,
    BackgroundSkyboxMaterial,
    Camera,
    Canvas,
    Controller,
    DirectionalLight,
    EventType,
    GfxGroup,
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
    UIRenderer,
    Viewport,
    call_later,
    display_jupyter_widget,
    get_app,
    linalg,
    qcall_later,
    run,
)
from fury.optpkg import optional_package
from fury.ui import UI, UIContext

cv2, have_cv2, _ = optional_package(
    "cv2",
    trip_msg=(
        "OpenCV has to be installed to record animations as mp4. "
        "Install it with `pip install fury[optional]`."
    ),
)


class Scene(GfxGroup):
    """
    Scene class to hold the actors in the scene.

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
        """
        Arrange the logical and spatial representation of actors.

        This class acts as a scene graph container, managing actors,
        background, and lighting for rendering.
        """
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
        """
        Create a skybox background actor from a cubemap texture.

        Parameters
        ----------
        cube_map : Texture
            A PyGfx Texture object (cubemap).

        Returns
        -------
        Background
            A PyGfx Background object configured with the skybox material.
        """
        return Background(
            geometry=None, material=BackgroundSkyboxMaterial(map=cube_map)
        )

    @property
    def background(self):
        """
        Get the background color of the scene.

        Returns
        -------
        tuple
            The current background color as an (R, G, B, A) tuple.
        """
        return self._bg_color

    @background.setter
    def background(self, value):
        """
        Set the background color of the scene.

        This replaces the current background actor (color or skybox)
        with a new uniform color background.

        Parameters
        ----------
        value : tuple
            The desired background color as an (R, G, B, A) tuple.
        """
        self.remove(self._bg_actor)
        self._bg_color = value
        self._bg_actor = Background.from_color(value)
        self.add(self._bg_actor)

    def set_skybox(self, cube_map):
        """
        Set a skybox as the scene background using a cubemap texture.

        This replaces the current background actor (color or skybox)
        with a new skybox background.

        Parameters
        ----------
        cube_map : Texture
            A PyGfx Texture object (cubemap) for the skybox.
        """
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
        """
        Add actors or UI elements to the scene.

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
        """
        Remove actors or UI elements from the scene.

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
    """
    Define an independent viewport within the window.

    Holds a scene graph, camera, and controller for rendering actors
    within a specific rectangular area of the window.
    """

    viewport: Viewport
    scene: Scene
    camera: Camera
    controller: Controller

    @property
    def size(self):
        """
        Get the size of the screen viewport.

        Returns
        -------
        tuple
            The width and height (w, h) of the viewport in pixels.
        """
        return self.viewport.rect[2:]

    @property
    def position(self):
        """
        Get the position of the screen viewport within the window.

        Returns
        -------
        tuple
            The x and y coordinates (x, y) of the viewport's top-left corner.
        """
        return self.viewport.rect[:2]

    @property
    def bounding_box(self):
        """
        Get the bounding box of the screen viewport within the window.

        Returns
        -------
        tuple
            The position and size (x, y, w, h) of the viewport.
        """
        return self.viewport.rect

    @bounding_box.setter
    def bounding_box(self, value):
        """
        Set the bounding box of the screen viewport within the window.

        Parameters
        ----------
        value : tuple
            The desired position and size (x, y, w, h) for the viewport.
        """
        self.viewport.rect = value


def add_ui_to_scene(ui_scene, ui_obj):
    """
    Recursively traverse and add UI hierarchy to the UI scene.

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
    """
    Recursively traverse and remove UI hierarchy from the UI scene.

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
    """
    Compose a Screen object with viewport, scene, camera, and controller.

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
        A configured Screen object ready for rendering.
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
        controller = TrackballController(camera, register_events=vp)

    screen = Screen(vp, scene, camera, controller)
    update_camera(camera, screen.size, scene)
    return screen


def update_camera(camera, size, target):
    """
    Update the camera's view to encompass the target object or scene.

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
        The PyGfx object or scene the camera should focus on.
    """
    if isinstance(target, Scene):
        target = target.main_scene

    if (isinstance(target, GfxScene) and len(target.children) > 3) or (  # type: ignore [misc]
        not isinstance(target, GfxScene) and target is not None  # type: ignore [misc]
    ):
        camera.show_object(target)
    elif size is not None:
        camera.width = size[0]
        camera.height = size[1]


def _get_scene_center(camera, scene):
    """
    Center of scene using the bounding box.

    Parameters
    ----------
    camera : Camera
        The camera object looking at the scene.
    scene : Scene
        Scene used for calculating the center.
        If the scene is empty, the center is calculated based on the camera position and
        forward direction.

    Returns
    -------
    ndarray
        The center of the scene as a 3D numpy array.
    """
    bbox = scene.main_scene.get_world_bounding_box()
    if bbox is not None and np.isfinite(bbox).all():
        return np.asarray(0.5 * (bbox[0] + bbox[1]), dtype=np.float32)

    camera_pos = np.asarray(camera.world.position, dtype=np.float32)
    return camera_pos + np.asarray(camera.world.forward, dtype=np.float32)


def _reference_up_for_axis(axis_dir):
    """
    Reference up position based on the axis direction.

    The reference up needs change to handle the camera alignment close to the fixed
    reference up.

    Parameters
    ----------
    axis_dir : tuple or ndarray
        The direction for which reference up is required.

    Returns
    -------
    ndarray
        The reference up according to the axis direction.
    """
    if axis_dir[1] > 0.9:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if axis_dir[1] < -0.9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def set_camera_from_axis(screen, axis_direction):
    """
    Set camera based on the axis direction proposed.

    This method will preserves the distance it actually had from the center of the
    scene. It will only move the camera to the new angle based on the direction.

    Parameters
    ----------
    screen : Screen
        Screen in which the camera needs to be updated.
    axis_direction : tuple or ndarray
        The axis direction to set the camera to from the center of the scene.
    """
    camera = screen.camera
    target = _get_scene_center(camera, screen.scene)
    axis_direction = np.array(axis_direction, dtype=np.float32, copy=True)
    axis_direction /= np.linalg.norm(axis_direction)

    camera_pos = np.asarray(camera.world.position, dtype=np.float32)
    camera_forward = np.asarray(camera.world.forward, dtype=np.float32)
    distance = float(np.abs(np.dot(target - camera_pos, camera_forward)))
    if not np.isfinite(distance) or distance < 1e-6:
        distance = float(max(getattr(camera, "depth", 1.0), 1.0))

    current_view_axis = camera_pos - target
    current_view_axis_norm = float(np.linalg.norm(current_view_axis))

    if current_view_axis_norm > 1e-6:
        current_view_axis /= current_view_axis_norm
        if float(np.dot(current_view_axis, axis_direction)) > 0.995:
            axis_direction *= -1.0

    new_pos = target + axis_direction * distance
    camera.world.reference_up = _reference_up_for_axis(axis_direction)
    camera.world.position = new_pos
    camera.look_at(target)
    update_camera(camera, None, screen.scene)


def update_viewports(screens, screen_bbs):
    """
    Update the bounding boxes and cameras of multiple screens.

    Parameters
    ----------
    screens : list of Screen
        The list of Screen objects to update.
    screen_bbs : list of tuple
        A list of bounding boxes (x, y, w, h), one for each screen in `screens`.
    """
    for screen, screen_bb in zip(screens, screen_bbs, strict=False):
        screen.bounding_box = screen_bb
        update_camera(screen.camera, screen.size, screen.scene)


def render_screens(renderer, screens, stats=None, is_dirty=False):
    """
    Render multiple screens within a single renderer update cycle.

    Parameters
    ----------
    renderer : Renderer
        The PyGfx Renderer object to draw into.
    screens : list of Screen
        The list of Screen objects to render.
    stats : Stats, optional
        Stats helper to display FPS overlay.
    is_dirty : bool, optional
        If True, triggers layout recalculations for UI elements.
    """
    if stats is not None:
        stats.start()

    for screen in screens:
        scene_root = screen.scene

        if is_dirty:
            for ui_element in scene_root.ui_elements:
                if hasattr(ui_element, "update_layout"):
                    ui_element.update_layout()

        screen.viewport.render(scene_root.main_scene, screen.camera, flush=False)
        screen.viewport.render(scene_root.ui_scene, scene_root.ui_camera, flush=False)

    if stats is not None:
        stats.stop()
        stats.render(flush=False)

    renderer.flush()


def reposition_ui(screens):
    """
    Update the positions of all UI elements across multiple screens.

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
    """
    Calculate screen bounding boxes based on a layout configuration.

    The `screens` list defines vertical sections, and each element within
    specifies the number of horizontal sections in that vertical column.

    Parameters
    ----------
    screens : list of int or list of tuple or None
        Layout configuration.
        If a list of integers is provided, each integer represents a vertical column
        and specifies the number of horizontal rows within it based on the size.
        If a list of tuples is provided, and each tuple has 4 elements,
        they are treated as explicit bounding boxes (x, y, w, h) for each screen.
        regardless of the `size` parameter.
        If None or empty, assumes a single screen covering the full size.
    size : tuple
        The total size (width, height) of the window or area to divide.

    Returns
    -------
    list of tuple
        A list of calculated bounding boxes (x, y, w, h) for each screen.
    """
    if screens is None or not screens:
        return [(0, 0, *size)]

    if all(isinstance(screen, (tuple, list)) for screen in screens):
        if all(len(screen) == 4 for screen in screens):
            return screens
        else:
            logging.error("Invalid screen bounding box format. Expected (x, y, w, h).")
            sys.exit(1)

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
    """
    Show manager for the rendering window.

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
    pixel_ratio : float, optional
        The ratio between render buffer and display buffer pixels. If None,
        the display's native ratio is used. Avoid fractional values,
        which resample screen-space text and soften its strokes.
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
    imgui : bool, optional
            Whether to enable ImGui UI rendering support.
    imgui_draw_function : callable, optional
        A function that updates the ImGui UI elements each frame.
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
        pixel_ratio=None,
        camera_light=True,
        screen_config=None,
        enable_events=True,
        qt_app=None,
        qt_parent=None,
        show_fps=False,
        max_fps=60,
        imgui=False,
        imgui_draw_function=None,
    ):
        """
        Manage the rendering window, scenes, and interactions.

        Handles window creation, screen layout, rendering loop, and
        event handling.
        """
        self._size = size
        if not title:
            title = "FURY 2.0"
        self._title = title
        self._is_qt = False
        self._qt_app = qt_app
        self._qt_parent = qt_parent
        self._is_initial_resize = None
        self._show_fps = show_fps
        self._max_fps = max_fps
        self._frame_count = 0
        self._window_type = self._setup_window(window_type)
        self._is_dragging = False
        self._drag_target = None

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer
        if pixel_ratio is not None:
            self.renderer.pixel_ratio = pixel_ratio
        self.renderer.add_event_handler(
            lambda event: self._resize(size=(event.width, event.height)),
            EventType.RESIZE,
        )
        self.renderer.add_event_handler(
            self._register_drag,
            EventType.POINTER_DOWN,
            EventType.POINTER_UP,
            EventType.POINTER_MOVE,
        )
        self.renderer.add_event_handler(
            self._handle_key_event,
            EventType.KEY_DOWN,
            EventType.KEY_UP,
        )

        self._total_screens = 0
        self._screen_config = screen_config
        self._calculate_total_screens()
        self._screen_setup(scene, camera, controller, camera_light)
        self.screens = self._create_screens()
        self._callbacks = {}
        self._animations = []

        self._stats = None
        self._stats_initialized = False

        self._imgui = None
        if imgui:
            self.enable_imgui(imgui_draw_function=imgui_draw_function)

        self.enable_events = enable_events
        self._on_resize = lambda _size: None
        self._resize(self._size)

    def _handle_drag(self, event):
        """
        Handle drag events for pointer interactions.

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

    def _toggle_screen_controllers(self, disable):
        """
        Toggle the enabled state for controllers across multiple screen viewports.

        Parameters
        ----------
        disable : bool
            If True, deactivates the screen controllers; if False, enables them.
        """
        for screen in self.screens:
            screen.controller.enabled = not disable

    def _register_drag(self, event):
        """
        Handle global pointer events (drag) at the renderer level.

        Parameters
        ----------
        event : PointerEvent
            The PyGfx pointer event object.
        """
        if event.type == EventType.POINTER_DOWN:
            self._is_dragging = True
            self._drag_target = event.target
            if UIContext.hot_ui:
                self._toggle_screen_controllers(disable=True)
        elif event.type == EventType.POINTER_UP:
            self._is_dragging = False
            self._drag_target = None
            self._toggle_screen_controllers(disable=False)
        elif event.type == EventType.POINTER_MOVE and self._is_dragging:
            self._handle_drag(event)

    def _handle_key_event(self, event):
        """
        Handle global keyboard events at the renderer level.

        Parameters
        ----------
        event : KeyboardEvent
            The PyGfx keyboard event object.
        """
        if not UIContext.active_ui:
            return

        if event.type == EventType.KEY_DOWN:
            UIContext.active_ui.on_key_press(event)

            self._repeat_key_event = event
            call_later(0.6, self._do_key_repeat, event)

        elif event.type == EventType.KEY_UP:
            UIContext.active_ui.on_key_release(event)
            repeat_event = getattr(self, "_repeat_key_event", None)
            if repeat_event and repeat_event.key == event.key:
                self._repeat_key_event = None

    def _do_key_repeat(self, target_event):
        """
        Timer callback to repeatedly dispatch a held key.

        Parameters
        ----------
        target_event : Event
            The specific key down event that triggered this repeating timer.
        """
        if getattr(self, "_repeat_key_event", None) is not target_event:
            return

        if not UIContext.active_ui:
            self._repeat_key_event = None
            return

        UIContext.active_ui.on_key_press(target_event)

        call_later(0.05, self._do_key_repeat, target_event)

    def _screen_setup(self, scene, camera, controller, camera_light):
        """
        Prepare scene, camera, controller, and light lists for screen creation.

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
            Input camera light configuration.
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
        """
        Initialize the appropriate canvas window based on the type.

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
            If an invalid `window_type` is provided.
        """
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
        """
        Create all Screen objects based on the prepared configurations.

        Returns
        -------
        list of Screen
            The list of created Screen objects.
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

    def _resize(self, size):
        """
        Handle window resize events by updating viewports and re-rendering.

        Parameters
        ----------
        size : tuple
            The size (width, height) of the window in pixels.
        """
        self._on_resize(size)
        UIContext.canvas_size = size
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )
        reposition_ui(self.screens)
        self.render()

    def _on_repeat_callback(self, func, time, name, *args):
        """
        Internal method to handle the timing and execution of callbacks.

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
        """
        Register a callback function to be called after a time interval.

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
        """
        Cancel a registered callback by its name.

        Parameters
        ----------
        name : str
            The unique name of the callback to cancel.
        """
        if name in self._callbacks:
            del self._callbacks[name]

    def resize_callback(self, func):
        """
        Set a callback function to be called on window resize events.

        Parameters
        ----------
        func : callable
            A function that takes a single argument (size tuple) and is called
            whenever the window is resized.
        """
        self._on_resize = func

    def cancel_resize_callback(self):
        """Cancel the window resize callback function."""
        self._on_resize = lambda _size: None

    def _setup_camera_animations(self, animation, camera, *, force=False):
        """
        Recursively set camera on CameraAnimation objects if needed.

        Parameters
        ----------
        animation : Animation or Timeline
            Animation object to inspect for camera animations.
        camera : pygfx.Camera
            Camera to assign to camera animations.
        force : bool, optional
            If True, replace existing cameras and return previous values.

        Returns
        -------
        list[tuple[CameraAnimation, Camera]]
            Camera animations that were modified and their previous cameras.
        """
        camera_changes = []
        if isinstance(animation, CameraAnimation) and (
            force or animation.camera is None
        ):
            camera_changes.append((animation, animation.camera))
            animation.camera = camera

        for child_animation in getattr(animation, "_animations", []):
            camera_changes.extend(
                self._setup_camera_animations(child_animation, camera, force=force)
            )
        return camera_changes

    def _update_animation(self, animation):
        """
        Update an animation or timeline and request a redraw.

        Parameters
        ----------
        animation : Animation or Timeline
            Animation object to update.
        """
        if isinstance(animation, Timeline):
            animation.update()
        else:
            animation.update_animation()
        self.render()

    def add_animation(self, animation, *, update_rate=1 / 60):
        """
        Add an animation or timeline to the render update loop.

        Parameters
        ----------
        animation : Animation or Timeline
            The animation or timeline to add and update during rendering.
        update_rate : float, optional
            The time interval in seconds between updates. Default is 1/60.
        """
        if not isinstance(animation, (Animation, Timeline)):
            raise TypeError("Expected an Animation or Timeline object.")

        if animation in self._animations:
            return

        self._animations.append(animation)
        animation._set_record_callback(self.record_animation)
        if self.screens:
            scene = self.screens[0].scene
            animation.add_to_scene(scene)
            self._setup_camera_animations(animation, self.screens[0].camera)
            if isinstance(animation, Timeline):
                animation.play()

        callback_name = f"animation_{id(animation)}"
        self.register_callback(
            self._update_animation,
            update_rate,
            True,
            callback_name,
            animation,
        )

    def remove_animation(self, animation):
        """
        Remove an animation or timeline from the render update loop.

        Parameters
        ----------
        animation : Animation or Timeline
            Animation object to remove.
        """
        if animation not in self._animations:
            return

        self._animations.remove(animation)
        self.cancel_callback(f"animation_{id(animation)}")
        if animation._record_callback == self.record_animation:
            animation._set_record_callback(None)
        if self.screens:
            animation.remove_from_scene(self.screens[0].scene)

    def record_animation(
        self, animation, fname, *, fps=30, speed=1.0, size=None, return_frames=False
    ):
        """
        Record an animation or timeline to an mp4 file.

        Parameters
        ----------
        animation : Animation or Timeline
            The animation or timeline to record.
        fname : str
            The output file name. The ``.mp4`` extension is added when missing.
        fps : int, optional
            The number of frames per second in the output video.
        speed : float, optional
            Playback speed multiplier used while sampling the animation.
        size : tuple[int, int], optional
            The offscreen render size as ``(width, height)``. If None, the show
            manager's current size is used.
        return_frames : bool, optional
            If True, return the recorded RGBA frames. Defaults to False to avoid
            storing long recordings in memory.

        Returns
        -------
        list[ndarray] or None
            The recorded RGBA frames when ``return_frames`` is True, otherwise None.
        """
        if not isinstance(animation, (Animation, Timeline)):
            raise TypeError("Expected an Animation or Timeline object.")
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        if speed <= 0:
            raise ValueError("speed must be greater than 0.")
        if not have_cv2:
            raise ImportError(
                "OpenCV has to be installed to record animations as mp4. "
                "Install it with `pip install fury[optional]`."
            )

        fname = str(fname)
        if not fname.endswith(".mp4"):
            fname += ".mp4"

        size = self._size if size is None else size
        scene = getattr(animation, "_scene", None)
        if scene is None and self.screens:
            scene = self.screens[0].scene

        show_m = ShowManager(
            scene=scene, size=size, window_type="offscreen", pixel_ratio=1.0
        )
        if getattr(animation, "_scene", None) is None:
            animation.add_to_scene(show_m.screens[0].scene)
        camera_changes = self._setup_camera_animations(
            animation, show_m.screens[0].camera, force=True
        )

        duration = animation.update_duration()
        timestamps = [0.0]
        if duration > 0:
            timestamps = np.arange(0, duration, speed / fps).tolist()

        frames = [] if return_frames else None
        writer = None
        timeline_state = None
        if isinstance(animation, Timeline):
            timeline_state = (animation.playing, animation.current_timestamp)
            if animation.playing:
                animation.pause()
        try:
            writer = cv2.VideoWriter(
                fname,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {fname!r}.")

            for timestamp in timestamps:
                if isinstance(animation, Timeline):
                    animation.seek(timestamp)
                else:
                    animation.update_animation(time=timestamp)
                render_screens(show_m.renderer, show_m.screens)
                frame = np.asarray(show_m.renderer.snapshot())
                if return_frames:
                    frames.append(frame)
                writer.write(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()
            for camera_animation, camera in camera_changes:
                camera_animation.camera = camera
            if timeline_state is not None:
                was_playing, current_timestamp = timeline_state
                animation.seek(current_timestamp)
                if was_playing:
                    animation.play()
            show_m.window.close()

        return frames

    def enable_imgui(self, *, imgui_draw_function=None):
        """
        Enable ImGui UI rendering support.

        Parameters
        ----------
        imgui_draw_function : callable, optional
            A function that updates the ImGui UI elements each frame.
            If None, no UI update function is set initially.
        """
        if self._imgui is None:
            self._imgui = UIRenderer(self.renderer.device, self.window)
            self.set_imgui_render_callback(imgui_draw_function)
        else:
            logging.warning("ImGui is already enabled for this ShowManager.")

    def disable_imgui(self):
        """Disable ImGui UI rendering support."""
        if self._imgui is not None:
            self._imgui = None
        else:
            logging.warning("ImGui is not enabled for this ShowManager.")

    def set_imgui_render_callback(self, imgui_draw_function):
        """
        Set the ImGui rendering callback function.

        Parameters
        ----------
        imgui_draw_function : callable
            A function that updates the ImGui UI elements each frame.
        """
        if not callable(imgui_draw_function):
            logging.warning("The provided ImGui draw function is not callable.")
            return

        if self._imgui is not None:
            self._imgui.set_gui(imgui_draw_function)
        else:
            logging.warning("ImGui is not enabled for this ShowManager.")

    def show_axes_gizmo(
        self,
        *,
        screen=0,
        size=30,
        thickness=2,
        position=None,
        labels=None,
        click_callback=None,
    ):
        """
        Add an axes helper to the first screen for orientation reference.

        Parameters
        ----------
        screen : int, optional
            Index of the screen whose viewport should host the gizmo.
            If None, defaults to 0 (the first screen). If the index is out of bounds,
            it will be clamped to the valid range of available screens.
        size : float, optional
            The length of the axes lines.
        thickness : float, optional
            The thickness of the axes lines.
        position : tuple, optional
            The (x, y) position of the axes helper in screen coordinates. If None,
            it defaults to (60, 60) pixels from the bottom-left corner.
            The position is relative to the screen's viewport, not the entire window.
            The origin is bottom-left of the screen viewport.
        labels : list of str, optional
            Custom labels for the axes.
            Defaults to ["-X", "+X", "-Y", "+Y", "-Z", "+Z"] if None.
        click_callback : callable, optional
            A function to be called when an axis disk or label is clicked. The function
            should accept a single argument, which will be the axis direction vector
            corresponding to the clicked axis.
        """
        if screen is None:
            logging.warning("Screen index is None. Defaulting to screen 0.")
            screen = 0
        elif isinstance(screen, int):
            if screen < 0:
                logging.warning(
                    f"Negative screen index {screen} is invalid. Defaulting to screen "
                    "0."
                )
                screen = 0
            elif screen >= len(self.screens):
                logging.warning(
                    f"Screen index {screen} exceeds available screens."
                    f" Defaulting to screen {len(self.screens) - 1}."
                )
                screen = len(self.screens) - 1
        else:
            logging.warning(
                f"Invalid screen index type: {type(screen)}. Expected int. Defaulting "
                "to screen 0."
            )
            screen = 0

        if position is not None:
            px, py = position
        else:
            px = py = 60

        if labels is None:
            labels = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]

        if click_callback is not None and callable(click_callback):
            self._axes_helper_click_callback = click_callback
        else:
            self._axes_helper_click_callback = lambda _axis_dir: None

        axes_helper_actors = create_axes_helper(labels=labels, thickness=thickness)
        self._axes_helper = axes_helper_actors.get("group", {})
        self._axes_helper_anchor = Group(name="Axes Helper Anchor")
        center_disk = axes_helper_actors.get("center_disk")
        axes_helper_disks = axes_helper_actors.get("disks", [])
        axes_helper_labels = axes_helper_actors.get("labels", [])
        axes_helper_lines = axes_helper_actors.get("lines", [])
        axis_vectors = [
            np.asarray(axis_vector, dtype=np.float32)
            for axis_vector in axes_helper_actors.get("axis_vectors", [])
        ]

        def _axes_pick_pointer_down(event):
            """
            Camera alignment callback for axes helper disk and label clicks.

            Parameters
            ----------
            event : PointerEvent
                The disk or label click event containing the target actor with the
                _axes_direction attribute set.
            """
            nonlocal camera_rotation, object_rotation
            axis_direction = np.asarray(event.target._axes_direction, dtype=np.float32)
            event.stop_propagation()
            set_camera_from_axis(self.screens[screen], axis_direction)
            camera_rotation = np.asarray([0, 0, 0, 1], dtype=np.float32).copy()
            object_rotation = np.asarray([0, 0, 0, 1], dtype=np.float32).copy()
            _axes_helper_render_callback()
            self._axes_helper_click_callback(axis_direction)

        for disk_actor, label_actor, axis_vector in zip(
            axes_helper_disks,
            axes_helper_labels,
            axis_vectors,
            strict=False,
        ):
            axis_direction = np.array(axis_vector, dtype=np.float32)
            disk_actor._axes_direction = axis_direction
            label_actor._axes_direction = axis_direction
            disk_actor.add_event_handler(
                _axes_pick_pointer_down, EventType.POINTER_DOWN
            )
            label_actor.add_event_handler(
                _axes_pick_pointer_down, EventType.POINTER_DOWN
            )

        self._axes_helper_anchor.local.position = [px, py, 0.5]
        self._axes_helper_anchor.local.scale = [size, size, 0.45]
        self._axes_helper_anchor.add(self._axes_helper)
        self.screens[screen].scene.ui_scene.add(self._axes_helper_anchor)

        camera = self.screens[screen].camera
        camera_rotation = camera.world.rotation
        object_rotation = self._axes_helper.local.rotation

        def _axes_helper_render_callback():
            """Update the axes helper according to camera rotation."""
            nonlocal camera_rotation, object_rotation

            r_delta = linalg.quat_mul(
                linalg.quat_inv(camera.world.rotation), camera_rotation
            )
            camera_rotation = camera.world.rotation

            self._axes_helper.local.rotation = linalg.quat_mul(
                linalg.quat_inv(r_delta), object_rotation
            )
            object_rotation = self._axes_helper.local.rotation
            inv_rotation = linalg.quat_inv(self._axes_helper.local.rotation)

            center_disk.local.rotation = inv_rotation
            for disk_actor, label_actor in zip(
                axes_helper_disks, axes_helper_labels, strict=False
            ):
                disk_actor.local.rotation = inv_rotation
                label_actor.local.rotation = inv_rotation

            cam_forward = camera.world.forward
            disk_depths = []
            for axis_vector in axis_vectors:
                pos = np.asarray(axis_vector, dtype=np.float32)
                disk_depths.append(float(np.dot(cam_forward, pos)))

            front_depth = min(disk_depths)
            back_depth = max(disk_depths)
            depth_span = max(back_depth - front_depth, 1e-6)
            min_disk_opacity = 0.1

            for disk_actor, label_actor, line_actor, depth in zip(
                axes_helper_disks,
                axes_helper_labels,
                axes_helper_lines,
                disk_depths,
                strict=False,
            ):
                depth_factor = (depth - front_depth) / depth_span
                disk_actor.opacity = 1.0 - depth_factor * (1.0 - min_disk_opacity)
                label_actor.opacity = 1.0 - depth_factor * (1.0 - min_disk_opacity)
                line_actor.opacity = 1.0 - depth_factor * (1.0 - min_disk_opacity)

        self.register_callback(
            _axes_helper_render_callback,
            time=0.016,
            repeat=True,
            name="axes_helper_rotation",
        )

    @property
    def app(self):
        """
        Get the associated QApplication instance, if any.

        Returns
        -------
        QApplication or None
            The QApplication instance if the window type is 'qt', otherwise None.
        """
        return self._qt_app

    @property
    def title(self):
        """
        Get the current window title.

        Returns
        -------
        str
            The text displayed in the window's title bar.
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set the window title.

        Parameters
        ----------
        value : str
            The desired text for the window's title bar.
        """
        self._title = value
        self.window.set_title(self._title)

    @property
    def pixel_ratio(self):
        """
        Get the current pixel ratio of the renderer.

        Returns
        -------
        float
            The ratio between render buffer and display buffer pixels.
        """
        return self.renderer.pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        """
        Set the pixel ratio of the renderer.

        Parameters
        ----------
        value : float
            The desired pixel ratio.
        """
        self.renderer.pixel_ratio = value

    @property
    def size(self):
        """
        Get the current size of the window.

        Returns
        -------
        tuple
            The current (width, height) of the window in logical pixels.
        """
        return self._size

    @property
    def callbacks(self):
        """
        Get the registered callbacks.

        This only returns the callbacks that are set to repeat.

        Returns
        -------
        dict
            A dictionary of registered callbacks with their names as keys.
        """
        return self._callbacks

    @property
    def imgui(self):
        """
        Get the ImGui UI renderer if enabled.

        Returns
        -------
        UIRenderer or None
            The UIRenderer instance if ImGui is enabled, otherwise None.
        """
        return self._imgui

    @property
    def device(self):
        """
        Get the underlying GPU device from the renderer.

        Returns
        -------
        wgpu.GPUDevice
            The GPU device used by the renderer for rendering operations.
        """
        return self.renderer.device

    def set_enable_events(self, value):
        """
        Enable or disable mouse and keyboard interactions for all screens.

        Parameters
        ----------
        value : bool
            Set to True to enable events, False to disable them.
        """
        self.enable_events = value
        if value:
            self.renderer.enable_events()
        else:
            self.renderer.disable_events()
        for s in self.screens:
            s.controller.enabled = value

    def get_fps(self):
        """
        Get the current FPS from the stats overlay if available.

        Returns
        -------
        int or None
            The current FPS value, or None if stats are not initialized or
            FPS has not been computed yet.
        """
        if self._stats is not None:
            return getattr(self._stats, "_fps", None)
        return None

    def snapshot(self, fname):
        """
        Save a snapshot of the current rendered content to a file.

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

        self._frame_count += 1
        update_layout = True if self._frame_count == 2 else False
        render_screens(
            self.renderer, self.screens, stats=self._stats, is_dirty=update_layout
        )
        self._imgui and self._imgui.render()
        self.window.request_draw()

    def render(self):
        """Request a redraw of all screens in the window."""
        if self._is_qt and self._qt_parent is not None:
            self._qt_parent.show()
        self.window.request_draw(self._draw_function)

    def start(self):
        """
        Start the rendering event loop and display the window.

        This call blocks until the window is closed, unless running in
        an offscreen or specific environment (like FURY_OFFSCREEN).
        """
        self.render()
        if "FURY_OFFSCREEN" in os.environ and os.environ["FURY_OFFSCREEN"].lower() in [
            "true",
            "1",
        ]:
            frames = []
            record_animation = os.environ.get("FURY_RECORD_ANIMATION", "").lower() in [
                "true",
                "1",
            ]
            if self._callbacks and record_animation:
                max_frames = 300
                env_max_frames = os.environ.get("FURY_OFFSCREEN_MAX_FRAMES")
                if env_max_frames and env_max_frames.isdigit():
                    max_frames = int(env_max_frames)

                for _ in range(max_frames):
                    if not self._callbacks:
                        break

                    for _name, cb_info in list(self._callbacks.items()):
                        func, cb_time, repeat, args = cb_info
                        func(*args)

                    self._draw_function()
                    arr = np.asarray(self.renderer.snapshot())
                    frames.append(image_from_array(arr))

            if frames:
                gif_name = f"{self._title.replace(' ', '_')}.gif"
                frames[0].save(
                    gif_name,
                    append_images=frames[1:],
                    save_all=True,
                    duration=30,
                    loop=0,
                )
            else:
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

        if self._window_type == "jupyter":
            display_jupyter_widget(self.window)

    def close(self):
        """
        Close the rendering window and terminate the application if necessary.
        """
        self.window.close()


def snapshot(
    *,
    scene=None,
    screen_config=None,
    fname="output.png",
    actors=None,
    return_array=False,
):
    """
    Take a snapshot using an offscreen window.

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
        Otherwise, returns None.
    """
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


def analyze_snapshot(im, *, colors=None, find_objects=True, strel=None):
    """
    Analyze snapshot from memory or file.

    Parameters
    ----------
    im : str or array
        If string then the image is read from a file otherwise the image is
        read from a numpy array. The array is expected to be of shape (X, Y, 3)
        or (X, Y, 4) where the last dimensions are the RGB or RGBA values.
    colors : tuple or list of tuples, optional
        List of colors to search in the image.
    find_objects : bool, optional
        If True it will calculate the number of objects that are different
        from the background and return their position in a new image.
    strel : 2d array, optional
        Structure element to use for finding the objects of size (3, 3).

    Returns
    -------
    ReportSnapshot
        This is an object with attributes like ``colors_found`` that give
        information about what was found in the current snapshot array ``im``.
    """
    if isinstance(im, str):
        im = load_image(im)

    class ReportSnapshot:
        """Report class for snapshot analysis results."""

        objects = None
        labels = None
        colors_found = False

        def __str__(self):
            """
            String method for printing.

            Returns
            -------
            str
                A formatted string report of the snapshot analysis.
            """
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
        for i, col in enumerate(colors):
            flags[i] = np.any(np.any(np.all(np.equal(im[..., :3], col[:3]), axis=-1)))

        report.colors_found = flags

    if find_objects is True:
        weights = [0.299, 0.587, 0.144]
        gray = np.dot(im[..., :3], weights)
        bg_color2 = im[0, 0][:3]
        background = np.dot(bg_color2, weights)

        if strel is None:
            strel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        labels, objects = ndimage.label(gray != background, strel)
        report.labels = labels
        report.objects = objects

    return report


def show(
    actors,
    *,
    window_type="default",
    title="FURY 2.0",
):
    """
    Display one or more actors in a new window quickly.

    A convenience function to quickly visualize actors without manually
    setting up a Scene or ShowManager.

    Parameters
    ----------
    actors : Object or list of Object
        The PyGfx actor(s) to display.
    window_type : str, optional
        The type of window canvas to create ('default', 'glfw', 'qt',
        'jupyter', 'offscreen').
    title : str, optional
        The title for the window.
    """
    scene = Scene()
    if isinstance(actors, Iterable):
        scene.add(*actors)
    else:
        scene.add(actors)
    show_m = ShowManager(scene=scene, window_type=window_type, title=title)
    show_m.start()
