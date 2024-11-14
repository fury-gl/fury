from functools import reduce

from PIL.Image import fromarray as image_from_array
from numpy import asarray as np_asarray
from pygfx import WgpuRenderer
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.v2.window import (
    Scene,
    calculate_screen_sizes,
    create_screen,
    render_screens,
    update_viewports,
)


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
        window_type="auto",
        controller_style="orbit",
        pixel_ratio=1,
        camera_light=True,
        screen_config=None,
    ):
        self.size = size
        self._title = title
        self._setup_window(window_type)

        if renderer is None:
            renderer = WgpuRenderer(self.window)
        self.renderer = renderer

        self._screen_config = screen_config
        self._total_screens = 0
        self._calculate_total_screens()

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

        self.screens = self._create_screens()
        update_viewports(
            self.screens,
            calculate_screen_sizes(self._screen_config, self.renderer.logical_size),
        )

        self.png_magnify = png_magnify
        self.order_transparent = order_transparent
        self.stereo = stereo
        self.timers = []
        self._fps = 0
        self._last_render_time = 0

        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.blend_mode = "weighted_plus"

        self.renderer.enable_events()

        self.renderer.add_event_handler(self.resize, "resize")

    def _setup_window(self, window_type):
        if window_type == "auto":
            self.window = WgpuCanvas(size=self.size, title=self._title)
        elif window_type == "jupyter":
            self.window = JupyterWgpuCanvas(size=self.size, title=self._title)
        elif window_type == "offscreen":
            self.window = OffscreenWgpuCanvas(size=self.size, title=self._title)
        else:
            self.window = WgpuCanvas(size=self.size, title=self._title)

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

    @title.setter
    def title(self, value):
        self.renderer.pixel_ratio = value

    def snapshot(self, fname):
        img = image_from_array(np_asarray(self.renderer.snapshot()))
        img.save(fname)

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


def record(*, scene=None, screen_config=None, fname="output.png", actors=None):
    if actors is not None:
        scene = Scene()
        scene.add(*actors)

    show_m = ShowManager(
        scene=scene, screen_config=screen_config, window_type="offscreen"
    )
    show_m.render()
    show_m.window.draw()
    show_m.snapshot(fname)


def display(*, actors):
    scene = Scene()
    scene.add(*actors)
    show_m = ShowManager(scene=scene)
    show_m.start()
