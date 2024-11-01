import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.decorators import warn_on_args_to_kwargs


class ShowManager:

    @warn_on_args_to_kwargs
    def __init__(
        self,
        *,
        scene=None,
        camera=None,
        title="FURY 2.0",
        size=(300, 300),
        png_magnify=1,
        reset_camera=True,
        order_transparent=False,
        stereo="off",
        multi_samples=8,
        max_peels=4,
        occlusion_ratio=0.0,
        window_type="auto",
        controller_style="orbit",
        pixel_ratio=1
    ):

        if scene is None:
            scene = gfx.Scene()
        if camera is None:
            camera = gfx.OrthographicCamera(70, 16 / 9)

        self.scene = scene
        self._title = title
        self.camera = camera
        self.size = size
        self.png_magnify = png_magnify
        self.reset_camera = reset_camera
        self.order_transparent = order_transparent
        self.stereo = stereo
        self.timers = []
        self._fps = 0
        self._last_render_time = 0

        if window_type == "auto":
            self.window = WgpuCanvas(size=self.size, title=title)
        elif window_type == "jupyter":
            self.window = JupyterWgpuCanvas(size=self.size, title=title)
        elif window_type == "offscreen":
            self.window = OffscreenWgpuCanvas(size=self.size, title=title)
        else:
            self.window = WgpuCanvas(size=self.size, title=title)
            # self.window.set_

        self.renderer = gfx.renderers.WgpuRenderer(self.window)
        self.renderer.pixel_ratio = pixel_ratio

        if controller_style == "orbit":
            pass

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
