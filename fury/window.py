from dataclasses import dataclass
from functools import reduce

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
        super().__init__()

        self._bg_color = background
        self._bg_actor = None

        if skybox:
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
        return Background(
            geometry=None, material=BackgroundSkyboxMaterial(map=cube_map)
        )

    @property
    def background(self):
        return self._background_color

    @background.setter
    def background(self, value):
        self.remove(self._bg_actor)
        self._bg_actor = Background.from_color(value)
        self.add(self._bg_actor)

    def set_skybox(self, cube_map):
        self.remove(self._bg_actor)
        self._bg_actor = self._skybox(cube_map)
        self.add(self._bg_actor)

    def clear(self):
        self.remove(*self.children)


@dataclass
class Screen:
    viewport: Viewport
    scene: Scene
    camera: Camera
    controller: Controller

    @property
    def size(self):
        return self.viewport.rect[2:]

    @property
    def position(self):
        return self.viewport.rect[:2]

    @property
    def bounding_box(self):
        return self.viewport.rect

    @bounding_box.setter
    def bounding_box(self, value):
        self.viewport.rect = value


def create_screen(
    renderer, *, rect=None, scene=None, camera=None, controller=None, camera_light=True
):
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
    camera.width = size[0]
    camera.height = size[1]

    if (isinstance(target, Scene) and len(target.children) > 3) or not isinstance(
        target, Scene
    ):
        camera.show_object(target)


def update_viewports(screens, screen_bbs):
    for screen, screen_bb in zip(screens, screen_bbs):
        screen.bounding_box = screen_bb
        update_camera(screen.camera, screen.size, screen.scene)


def render_screens(renderer, screens):
    for screen in screens:
        screen.viewport.render(screen.scene, screen.camera, flush=False)

    renderer.flush()


def calculate_screen_sizes(screens, size):
    if screens is None:
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
        png_magnify=1,
        order_transparent=False,
        stereo="off",
        multi_samples=8,
        max_peels=4,
        occlusion_ratio=0.0,
        blend_mode="weighted_plus",
        window_type="auto",
        pixel_ratio=1,
        camera_light=True,
        screen_config=None,
        enable_events=True,
    ):
        self.size = size
        self._title = title
        self._setup_window(window_type)

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer
        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.blend_mode = blend_mode
        self.renderer.add_event_handler(self.resize, "resize")

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

        self.png_magnify = png_magnify
        self.order_transparent = order_transparent
        self.stereo = stereo
        self.timers = []
        self._fps = 0
        self._last_render_time = 0

    def _screen_setup(self, scene, camera, controller, camera_light):
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
        if window_type == "auto":
            self.window = Canvas(size=self.size, title=self._title)
        elif window_type == "jupyter":
            self.window = JupyterCanvas(size=self.size, title=self._title)
        elif window_type == "offscreen":
            self.window = OffscreenCanvas(size=self.size, title=self._title)
        else:
            self.window = Canvas(size=self.size, title=self._title)

    def _calculate_total_screens(self):
        if self._screen_config is None:
            self._total_screens = 1
        elif isinstance(self._screen_config[0], int):
            self._total_screens = reduce(lambda a, b: a + b, self._screen_config)
        else:
            self._total_screens = len(self._screen_config)

    def _create_screens(self):
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

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.window.set_title(self._title)

    @property
    def pixel_ratio(self):
        return self.renderer.pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        self.renderer.pixel_ratio = value

    @property
    def enable_events(self):
        return self._enable_events

    @enable_events.setter
    def enable_events(self, value):
        self._enable_events = value
        if value:
            self.renderer.enable_events()
        else:
            self.renderer.disable_events()
        for s in self.screens:
            s.controller.enabled = value

    def snapshot(self, fname):
        arr = np.asarray(self.renderer.snapshot())
        img = image_from_array(arr)
        img.save(fname)
        return arr

    def render(self):
        self.window.request_draw(lambda: render_screens(self.renderer, self.screens))

    def start(self):
        self.render()
        run()

    def resize(self, _event):
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )
        self.render()


def snapshot(
    *,
    scene=None,
    screen_config=None,
    fname="output.png",
    actors=None,
    return_array=False,
):
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


def display(*, actors):
    scene = Scene()
    scene.add(*actors)
    show_m = ShowManager(scene=scene)
    show_m.start()
