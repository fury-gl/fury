import numpy as np
from viz_shader_canvas import cube, disk, rectangle, rectangle2, square
import vtk

from fury import actor, window

scene = window.Scene()
# scene.projection('parallel')
scene.add(actor.axes())
# scene.background((1, 1, 1))
showm = window.ShowManager(
    scene, size=(1920, 1080), order_transparent=True, interactor_style='custom'
)

obj = 'square'

if obj == 'square':

    canvas_actor = square()
    canvas_actor.GetProperty().BackfaceCullingOff()
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()

if obj == 'rectangle':

    canvas_actor = rectangle(size=(100, 100))
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()

if obj == 'rectangle2':
    n_points = 3
    # centers = np.random.rand(n_points, 3)
    # print(centers)
    centers = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 0]])
    colors = 255 * np.random.rand(n_points, 3)
    canvas_actor = rectangle2(centers=centers, colors=colors)
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()
    mapper.MapDataArrayToVertexAttribute(
        'my_centers', 'Points', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1
    )

if obj == 'cube':

    # rec.SetPosition(100, 0, 0)
    canvas_actor = cube()
    scene.add(canvas_actor)
    scene.background((1, 1, 1))
    # window.show(scene)
    mapper = canvas_actor.GetMapper()

if obj == 'disk':

    canvas_actor = disk()
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()


mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Dec',
    True,
    """
    //VTK::Light::Dec
    uniform float time;
    """,
    False,
)

global timer
timer = 0


def timer_callback(obj, event):

    global timer
    timer += 1.0
    # print(timer)
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
    vtk.vtkShader.Vertex,
    '//VTK::Normal::Dec',
    True,
    """
    //VTK::Normal::Dec
    out vec4 myVertexMC;
    in vec3 my_centers[3]; // now declare our attribute
    out vec3 my_centers_out[3];
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
    my_centers_out = my_centers;
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
    varying vec3  my_centers_out[3];
    """,
    False,
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    '//VTK::Light::Impl',
    True,
    """
    //VTK::Light::Impl
    if(myVertexMC == vec4(0.5, 0.5, 0.5, 1.0)) {
        fragOutput0 = vec4(1., 1., 1., 1.);
        return;
    }
    vec3 rColor = vec3(.9, .0, .3);
    vec3 gColor = vec3(.0, .9, .3);
    vec3 bColor = vec3(.0, .3, .9);
    vec3 yColor = vec3(.9, .9, .3);

    float tm = .2; // speed
    float vcm = 5;

    float a = sin(myVertexMC.y * vcm - time * tm) / 2.;
    float b = cos(myVertexMC.y * vcm - time * tm) / 2.;
    float c = sin(myVertexMC.y * vcm - time * tm + 3.14) / 2.;
    float d = cos(myVertexMC.y * vcm - time * tm + 3.14) / 2.;

    float div = 0.01; // default 0.01

    float e = div / abs(myVertexMC.x + a);
    float f = div / abs(myVertexMC.x + b);
    float g = div / abs(myVertexMC.x + c);
    float h = div / abs(myVertexMC.x + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;
    fragOutput0 = vec4(destColor, 1.);
    //fragOutput0 = vec4(1 - myVertexMC.x, 1 - myVertexMC.y, 0, 1.);
    //fragOutput0 = vec4(myVertexMC.x, 0, 0, 1.);
    //vec2 p = vertexVCVSOutput.xy; //- vec2(1.5,0.5);
    vec2 p = myVertexMC.xy;
    fragOutput0 = vec4(p, 0., 1.);
    //bc
    //float z = 1.0 - length(p)/0.05;
    //if (z < 0.0) {fragOutput0 = vec4(0., 1., 0., 1.);return;}
    //fragOutput0 = vec4(1., 0., 0., 1.);
    if(length(p - vec2(0, 0)) < 0.2) {
        fragOutput0 = vec4(1, 0., 0., 1.);
    }

    if(length(p - vec2(1, 1)) < 0.2) {
        fragOutput0 = vec4(1, 0., 0., 1.);
    }
    """,
    False,
)


# showm.add_timer_callback(True, 100, timer_callback)


showm.start()
