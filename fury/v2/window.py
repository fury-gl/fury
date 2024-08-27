from PIL import Image
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas
import pygfx as gfx
import numpy as np


class ShowManager:

    def __init__(
            self,
            scene=None,
            camera=None,
            is_jupyter=False,
            is_offscreen=False
    ):

        self._canvas = None
        if is_jupyter:
            self._canvas = JupyterWgpuCanvas()
        elif is_offscreen:
            self._canvas = OffscreenWgpuCanvas()
        else:
            self._canvas = WgpuCanvas()
        self._canvas.set_title("FURY 2.0")
        self._renderer = gfx.renderers.WgpuRenderer(self._canvas)

        if scene is None:
            scene = gfx.Scene()
        self._scene = scene
        self._scene.add(gfx.AmbientLight())

        if camera is None:
            camera = gfx.PerspectiveCamera()
        self._camera = camera

        gfx.OrbitController(self._camera, register_events=self._renderer)
        self._renderer.enable_events()

    def render(self):
        self._canvas.request_draw(lambda: self._renderer.render(
            self._scene, self._camera
        ))

    def start(self):
        run()

    def snapshot(self, fname):
        img = Image.fromarray(np.asarray(self._renderer.snapshot()))
        img.save(fname)

    @property
    def scene(self): return self._scene

    @property
    def canvas(self): return self._canvas
