# Create a basic shader

import numpy as np
from fury import window, actor, ui, io, utils
import vtk

# import the 3D model of choice
# to import a model, use io.load_polydata()
# currently supported formats include OBJ, VKT, FIB, PLY, STL and XML

utah = io.load_polydata('models/utah.obj')
utah = utils.get_polymapper_from_polydata(utah)
utah = utils.get_actor_from_polymapper(utah)
mapper = utah.GetMapper()


# Replace fragment shader using vtkShader.Vertex
mapper.AddShaderReplacement(
     vtk.vtkShader.Vertex,
    "//VTK::Output::Dec", # declaration any uniforms/varying needed for normals
    True,
    """
    //VTK::Output::Dec
    out vec4 myVertexVC;

    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::Output::Impl",# implementation for normals
    True,
    """
    //VTK::Output::Impl
    myVertexVC = vertexVC;
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


# create a scene to be rendered in 
scene = window.Scene()

showm = window.ShowManager(scene, size=(1024, 720), reset_camera=False)

global timer
timer = 0


def timer_callback(obj, event):
    global timer
    timer += 1.0
    showm.render()
    #  print(timer)
    #  scene.azimuth(10)


#  add a decorator to your custom callback
@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf("time", timer)
        except ValueError:
            pass

# Add a textblock to add text to 
tb = ui.TextBlock2D()
tb.message = "Hello Shaders"

# change the property of the actor
utah.GetProperty().SetOpacity(0.5)

# Invoke callbacks to any VTK object
mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

showm.initialize()
showm.add_timer_callback(True, 30, timer_callback)
# add created actors to the scene
scene.add(utah)
scene.add(tb)

showm.start()