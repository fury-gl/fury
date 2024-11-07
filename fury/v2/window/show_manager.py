from functools import reduce

from PIL.Image import fromarray as image_from_array
from numpy import asarray as np_asarray
from pygfx import DirectionalLight, OrbitController, PerspectiveCamera
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.decorators import warn_on_args_to_kwargs
from fury.v2.window import Renderer, Screen


class ShowManager:
    @warn_on_args_to_kwargs()
    def __init__(
        self,
        *,
        renderer=None,
        scene=None,
        camera=None,
        title="FURY 2.0",
        size=(800, 800),
        png_magnify=1,
        reset_camera=True,
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
        if window_type == "auto":
            self.window = WgpuCanvas(size=self.size, title=title)
        elif window_type == "jupyter":
            self.window = JupyterWgpuCanvas(size=self.size, title=title)
        elif window_type == "offscreen":
            self.window = OffscreenWgpuCanvas(size=self.size, title=title)
        else:
            self.window = WgpuCanvas(size=self.size, title=title)

        if renderer is None:
            renderer = Renderer(self.window)
        self.renderer = renderer

        self.scene = scene
        self._screen_config = screen_config

        self._total_screens = 0

        if self._screen_config is None:
            self._total_screens = 1
        elif isinstance(self._screen_config[0], int):
            self._total_screens = reduce(lambda a, b: a + b, self._screen_config)
        else:
            self._total_screens = len(self._screen_config)

        if not isinstance(scene, list):
            self.scene = [scene] * self._total_screens

        self.renderer.screens = self._create_screens()
        self.renderer.update_screens(
            _calculate_screen_sizes(self._screen_config, self.renderer.logical_size)
        )
        self._title = title
        self.camera = camera
        self.png_magnify = png_magnify
        self.reset_camera = reset_camera
        self.order_transparent = order_transparent
        self.stereo = stereo
        self.timers = []
        self._fps = 0
        self._last_render_time = 0

        self.renderer.pixel_ratio = pixel_ratio
        self.renderer.blend_mode = "weighted_plus"

        # if controller_style == "orbit":
        #     OrbitController(self.camera, register_events=self.renderer)
        self.renderer.enable_events()

        self.renderer.add_event_handler(self.resize, "resize")

    def _create_screens(self):
        screens = []
        for i in range(self._total_screens):
            screens.append(Screen(self.renderer, scene=self.scene[i]))
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
        self.window.request_draw(lambda: self.renderer.render_screens())

    def start(self):
        self.render()
        run()

    def resize(self, event):
        self.renderer.update_screens(
            _calculate_screen_sizes(self._screen_config, self.renderer.logical_size)
        )
        self.render()


@warn_on_args_to_kwargs()
def record(*, renderer=None, scene=None, fname="output.png"):
    if renderer is None:
        show_m = ShowManager(scene=scene, window_type="offscreen")
        show_m.render()
        show_m.window.draw()
        show_m.snapshot(fname)
    else:
        show_m = ShowManager(renderer=renderer, window_type="offscreen")


def _calculate_screen_sizes(screens, size):
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
