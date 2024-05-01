from viz_shader_canvas import cube
import vtk

from fury import window

scene = window.Scene()
showm = window.ShowManager(scene, order_transparent=True)

canvas_actor = cube()
canvas_actor.GetProperty().BackfaceCullingOff()
scene.add(canvas_actor)
scene.background((1, 1, 1))
mapper = canvas_actor.GetMapper()

# Modify the vertex shader to pass the position of the vertex
mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::Normal::Dec',  # replace the normal block
    True,  # before the standard replacements
    """
    //VTK::Normal::Dec  // we still want the default
    out vec4 myVertexMC;
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,  # replace the normal block
    '//VTK::Normal::Impl',  # before the standard replacements
    True,  # we still want the default
    """
    //VTK::Normal::Impl
    myVertexMC = vertexMC;
    """,
    False,
)

# // Define varying and uniforms for the fragment shader here
mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Normal::Dec',  # // replace the normal block
    True,  # // before the standard replacements
    """
    //VTK::Normal::Dec  // we still want the default
    varying vec4 myVertexMC;
    uniform float time;
    """,
    False,  # // only do it once
)

global timer
timer = 0


def timer_callback(obj, event):
    global timer
    timer += 1.0
    showm.render()
    # scene.azimuth(10)


@window.vtk.calldata_type(window.vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    program = calldata
    global timer
    if program is not None:
        try:
            program.SetUniformf('time', timer)
        except ValueError:
            pass


mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,  # // in the fragment shader
    '//VTK::Light::Impl',  # // replace the light block
    False,  # // after the standard replacements
    """
    //VTK::Light::Impl  // we still want the default calc
    vec3 rColor = vec3(.9, .0, .3);
    vec3 gColor = vec3(.0, .9, .3);
    vec3 bColor = vec3(.0, .3, .9);
    vec3 yColor = vec3(.9, .9, .3);

    float yScale = .5;  // 1 / 2. Original
    float xScale = 2.5;  // 5. Original
    float shiftScale = .2;  // .2 Original

    float a = yScale * sin(myVertexMC.y * xScale - time * shiftScale);
    float b = yScale * cos(myVertexMC.y * xScale - time * shiftScale);
    float c = yScale * sin(myVertexMC.y * xScale - time * shiftScale + 3.14);
    float d = yScale * cos(myVertexMC.y * xScale - time * shiftScale + 3.14);

    float e = .01 / abs(myVertexMC.x + a);
    float f = .01 / abs(myVertexMC.x + b);
    float g = .01 / abs(myVertexMC.x + c);
    float h = .01 / abs(myVertexMC.x + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;

    fragOutput0 = vec4(destColor, .75);
    """,
    False,
)


showm.add_timer_callback(True, 100, timer_callback)
showm.start()
