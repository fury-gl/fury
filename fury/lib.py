from typing import TypeAlias

import jinja2
import pygfx as gfx
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas

from fury.optpkg import optional_package

jupyter_pckg_msg = (
    "You do not have jupyter-rfb installed. The jupyter widget will not work for "
    "you. Please install or upgrade jupyter-rfb using pip install -U jupyter-rfb"
)

jupyter_rfb, have_jupyter_rfb, _ = optional_package(
    "jupyter_rfb", trip_msg=jupyter_pckg_msg
)
if have_jupyter_rfb:
    from wgpu.gui.jupyter import WgpuCanvas as JupyterWgpuCanvas

qt_pckg_msg = (
    "You do not have any qt package installed. The qt window will not work for "
    "you. Please install or upgrade any of PySide6, PyQt6, PyQt5, PySide2 "
    "using pip install -U <QtPackageName>"
)

PySide6, have_py_side6, _ = optional_package("PySide6", trip_msg=qt_pckg_msg)
PyQt6, have_py_qt6, _ = optional_package("PyQt6", trip_msg=qt_pckg_msg)
PyQt5, have_py_qt5, _ = optional_package("PyQt5", trip_msg=qt_pckg_msg)

if have_py_side6 or have_py_qt6 or have_py_qt5:
    from wgpu.gui.qt import WgpuCanvas as QtWgpuCanvas, get_app

if have_py_side6:
    from PySide6 import QtWidgets

if have_py_qt6:
    from PyQt6 import QtWidgets

if have_py_qt5:
    from PyQt5 import QtWidgets

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
Material = gfx.Material
Mesh = gfx.Mesh
Points = gfx.Points
Line = gfx.Line

MeshBasicMaterial = gfx.MeshBasicMaterial
MeshPhongMaterial = gfx.MeshPhongMaterial
PointsMaterial = gfx.PointsMaterial
PointsGaussianBlobMaterial = gfx.PointsGaussianBlobMaterial
PointsMarkerMaterial = gfx.PointsMarkerMaterial

TextMaterial = gfx.TextMaterial
Text = gfx.Text
LineMaterial = gfx.LineMaterial
LineArrowMaterial = gfx.LineArrowMaterial
LineThinMaterial = gfx.LineThinMaterial
LineThinSegmentMaterial = gfx.LineThinSegmentMaterial
LineSegmentMaterial = gfx.LineSegmentMaterial
LineDebugMaterial = gfx.LineDebugMaterial

DirectionalLight = gfx.DirectionalLight
OrbitController = gfx.OrbitController
PerspectiveCamera = gfx.PerspectiveCamera
Renderer = gfx.WgpuRenderer
run = run
Canvas = WgpuCanvas
OffscreenCanvas = OffscreenWgpuCanvas
BaseShader = gfx.renderers.wgpu.BaseShader
LineShader = gfx.renderers.wgpu.shaders.lineshader.LineShader
ThinLineShader = gfx.renderers.wgpu.shaders.lineshader.ThinLineShader
PrimitiveTopology = wgpu.PrimitiveTopology
CullMode = wgpu.CullMode
Binding = gfx.renderers.wgpu.Binding
RenderMask = gfx.renderers.wgpu.RenderMask
Buffer = gfx.Buffer
register_wgpu_render_function = gfx.renderers.wgpu.register_wgpu_render_function
load_wgsl = gfx.renderers.wgpu.load_wgsl
loader = gfx.renderers.wgpu.shader.templating.loader
WorldObject = gfx.WorldObject
if have_jupyter_rfb:
    JupyterCanvas = JupyterWgpuCanvas
else:
    JupyterCanvas = jupyter_rfb
if have_py_side6 or have_py_qt6 or have_py_qt5:
    QtCanvas = QtWgpuCanvas
else:
    QtCanvas = PySide6
    QtWidgets = PySide6
    get_app = PySide6

loader.mapping["fury"] = jinja2.PackageLoader("fury.shaders.wgsl", ".")
