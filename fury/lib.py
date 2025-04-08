from typing import TypeAlias

import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.optpkg import optional_package

jupyter_pckg_msg = (
    "You do not have jupyter-rfb installed. The jupyter widget will not work for "
    "you. Please install or upgrade jupyter-rfb using pip install -U jupyter-rfb"
)

jupyter_rfb, have_jupyter_rfb, _ = optional_package(
    "jupyter-rfb", trip_msg=jupyter_pckg_msg
)

if have_jupyter_rfb:
    from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas

Texture = gfx.Texture
AmbientLight = gfx.AmbientLight
Background = gfx.Background
BackgroundSkyboxMaterial = gfx.BackgroundSkyboxMaterial

# Classes that needed to be written as types
Camera: TypeAlias = gfx.Camera
Controller: TypeAlias = gfx.Controller
Scene: TypeAlias = gfx.Scene
Viewport: TypeAlias = gfx.Viewport

Geometry = gfx.Geometry
Mesh = gfx.Mesh
Points = gfx.Points

MeshBasicMaterial = gfx.MeshBasicMaterial
MeshPhongMaterial = gfx.MeshPhongMaterial
PointsMaterial = gfx.PointsMaterial
PointsGaussianBlobMaterial = gfx.PointsGaussianBlobMaterial
PointsMarkerMaterial = gfx.PointsMarkerMaterial

DirectionalLight = gfx.DirectionalLight
OrbitController = gfx.OrbitController
PerspectiveCamera = gfx.PerspectiveCamera
Renderer = gfx.WgpuRenderer
run = run
Canvas = WgpuCanvas
OffscreenCanvas = OffscreenWgpuCanvas
if have_jupyter_rfb:
    JupyterCanvas = JupyterWgpuCanvas
else:
    JupyterCanvas = jupyter_rfb
