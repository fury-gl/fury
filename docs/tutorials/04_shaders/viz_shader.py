# -*- coding: utf-8 -*-
"""
=============
Varying Color
=============

This example shows how to use shaders to generate a shaded output. We will
demonstrate how to load polydata then use a custom shader calls to render
a custom shaded model.
First, a bunch of imports.

"""

from fury import window, ui, io, utils
import vtk

###############################################################################
# Let's download  and load the model

from fury.data.fetcher import fetch_viz_models, read_viz_models
fetch_viz_models()
model = read_viz_models('utah.obj')


###############################################################################
#
# Let's start by loading the polydata of choice.
# For this example we use the standard utah teapot model.
# currently supported formats include OBJ, VKT, FIB, PLY, STL and XML

utah = io.load_polydata(model)
utah = utils.get_polymapper_from_polydata(utah)
utah = utils.get_actor_from_polymapper(utah)
mapper = utah.GetMapper()


###############################################################################
# To change the default shader we add a shader replacement.
# Specify vertex shader using vtkShader.Vertex
# Specify fragment shader using vtkShader.Fragment


mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::ValuePass::Dec",
    True,
    """
    //VTK::ValuePass::Dec
    out vec4 myVertexVC;
    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::ValuePass::Impl",
    True,
    """
    //VTK::ValuePass::Impl
    myVertexVC = vertexMC;
    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::Light::Dec",
    True,
    """
    //VTK::Light::Dec
    uniform float time;
    varying vec4 myVertexVC;
    """,
    False
)


mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    """
    //VTK::Light::Impl
    vec2 iResolution = vec2(1024,720);
    vec2 uv = myVertexVC.xy/iResolution;
    vec3 col = 0.5 + 0.5 * cos((time/30) + uv.xyx + vec3(0, 2, 4));
    fragOutput0 = vec4(col, 1.0);
    """,
    False
)


###############################################################################
# Let's create a scene.

scene = window.Scene()

global timer
timer = 0

##############################################################################
# The timer will call this user defined callback every 30 milliseconds.


def timer_callback(obj, event):
    global timer
    timer += 1.0
    showm.render()
    scene.azimuth(5)


###############################################################################
# We can use a decorator to callback to the shader.


@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf("time", timer)
        except ValueError:
            pass


###############################################################################
# Let's add a textblock to the scene with a custom message

tb = ui.TextBlock2D()
tb.message = "Hello Shaders"

###############################################################################
# Change the property of the actor

utah.GetProperty().SetOpacity(0.5)

###############################################################################
# Invoke callbacks to any VTK object

mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent,
                   vtk_shader_callback)


###############################################################################
# Show Manager
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (1024, 720)
showm = window.ShowManager(scene, size=current_size, reset_camera=False)

showm.initialize()
showm.add_timer_callback(True, 30, timer_callback)

scene.add(utah)
scene.add(tb)

interactive = False
if interactive:
    showm.start()

window.record(showm.scene, size=current_size, out_path="viz_shader.png")
