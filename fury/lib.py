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

qt_pckg_msg = (
    "You do not have any qt package installed. The qt window will not work for "
    "you. Please install or upgrade any of PySide6, PyQt6, PyQt5, PySide2 "
    "using pip install -U <QtPackageName>"
)

PySide6, have_py_side6, _ = optional_package("PySide6", trip_msg=qt_pckg_msg)
PyQt6, have_py_qt6, _ = optional_package("PyQt6", trip_msg=qt_pckg_msg)
PyQt5, have_py_qt5, _ = optional_package("PyQt5", trip_msg=qt_pckg_msg)

if have_py_side6 or have_py_qt6 or have_py_qt5:
    from wgpu.gui.qt import WgpuCanvas as QtWgpuCanvas

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
if have_py_side6 or have_py_qt6 or have_py_qt5:
    QtCanvas = QtWgpuCanvas
else:
    QtCanvas = PySide6
    QtWidgets = PySide6
