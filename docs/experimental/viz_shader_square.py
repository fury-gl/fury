import numpy as np
from viz_shader_canvas import cube, square
import vtk

from fury import actor, window

scene = window.Scene()
# scene.add(actor.axes())
# scene.background((1, 1, 1))
showm = window.ShowManager(
    scene, size=(1920, 1080), order_transparent=True, interactor_style='custom'
)

obj = 'cube'

if obj == 'square':

    canvas_actor = square(3)
    canvas_actor.GetProperty().BackfaceCullingOff()
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()

if obj == 'cube':

    # rec.SetPosition(100, 0, 0)
    canvas_actor = cube()
    canvas_actor.GetProperty().BackfaceCullingOff()
    # cu.GetProperty().FrontfaceCullingOn()
    scene.add(canvas_actor)
    scene.add(actor.axes())
    scene.background((1, 1, 1))
    # window.show(scene)
    mapper = canvas_actor.GetMapper()

global timer
timer = 0


def timer_callback(obj, event):

    global timer
    timer += 1.0
    # print(object)
    # mapper.GetInput().ComputeBounds()
    # print(timer)
    showm.render()
    # cu.SetPosition(timer*0.01, 0, 0)
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
    vtk.vtkShader.Vertex,
    '//VTK::Normal::Dec',
    True,
    """
    //VTK::Normal::Dec
    uniform float time;
    out vec4 myVertexMC;
    mat4 rotationMatrix(vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;

        return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                    0.0,                                0.0,                                0.0,                                1.0);
    }

    vec3 rotate(vec3 v, vec3 axis, float angle) {
        mat4 m = rotationMatrix(axis, angle);
        return (m * vec4(v, 1.0)).xyz;
    }

    vec3 ax = vec3(1, 0, 0);

    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::Normal::Impl',
    True,
    """
    //VTK::Normal::Impl
    myVertexMC = vertexMC;
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::Light::Impl',
    True,
    """
    //VTK::Light::Impl
    myVertexMC.xyz = rotate(vertexMC.xyz, ax, time*0.01);
    vertexVCVSOutput = MCVCMatrix * myVertexMC;
    gl_Position = MCDCMatrix * myVertexMC;

    """,
    False,
)


mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Normal::Dec',
    True,
    """
    //VTK::Normal::Dec
    varying vec4 myVertexMC;
    uniform float time;
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    """
    //VTK::Light::Impl
    vec3 rColor = vec3(.9, .0, .3);
    vec3 gColor = vec3(.0, .9, .3);
    vec3 bColor = vec3(.0, .3, .9);
    vec3 yColor = vec3(.9, .9, .3);

    float tm = .2; // speed
    float vcm = 5;
    vec4 tmp = myVertexMC;

    float a = sin(tmp.y * vcm - time * tm) / 2.;
    float b = cos(tmp.y * vcm - time * tm) / 2.;
    float c = sin(tmp.y * vcm - time * tm + 3.14) / 2.;
    float d = cos(tmp.y * vcm - time * tm + 3.14) / 2.;

    float div = .01; // default 0.01

    float e = div / abs(tmp.x + a);
    float f = div / abs(tmp.x + b);
    float g = div / abs(tmp.x + c);
    float h = div / abs(tmp.x + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;
    fragOutput0 = vec4(destColor, 1.);

    vec2 p = tmp.xy;

    p = p - vec2(time * 0.005, 0.);

    if (length(p - vec2(0, 0)) < 0.2) {
        fragOutput0 = vec4(1, 0., 0., .5);
    }
    """,
    False,
)


showm.add_timer_callback(True, 100, timer_callback)


showm.start()
