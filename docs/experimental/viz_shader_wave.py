from viz_shader_canvas import cube, disk, rectangle, square
import vtk

from fury import actor, window

scene = window.Scene()
scene.add(actor.axes())
# scene.background((1, 1, 1))
showm = window.ShowManager(scene, size=(1920, 1080), order_transparent=True)

obj = 'square'

if obj == 'square':

    canvas_actor = square()
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()

if obj == 'rectangle':

    canvas_actor = rectangle(size=(100, 100))
    scene.add(canvas_actor)
    mapper = canvas_actor.GetMapper()

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
    scene.set_camera(focal_point=(0, 0, 0))
    scene.azimuth(10)


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

    float a = sin(gl_FragCoord.y * 0.01 * vcm - time * tm) / 2.;
    float b = cos(gl_FragCoord.y * 0.01 * vcm - time * tm) / 2.;
    float c = sin(gl_FragCoord.y * 0.01 * vcm - time * tm + 3.14) / 2.;
    float d = cos(gl_FragCoord.y * 0.01 * vcm - time * tm + 3.14) / 2.;

    float div = 0.01; // default 0.01

    float e = div / abs(gl_FragCoord.x * 0.01 + a);
    float f = div / abs(gl_FragCoord.x * 0.01 + b);
    float g = div / abs(gl_FragCoord.x * 0.01 + c);
    float h = div / abs(gl_FragCoord.x * 0.01 + d);

    vec3 destColor = rColor * e + gColor * f + bColor * g + yColor * h;
    fragOutput0 = vec4(destColor, 1.);
    //fragOutput0 = vec4(normalVCVSOutput.x, 0, 0, 1.);
    //fragOutput0 = vec4(normalVCVSOutput.x, normalVCVSOutput.y, 0, 1.);
    fragOutput0 = vec4(gl_FragCoord.x, 0, 0, 1.);
    """,
    False,
)


showm.add_timer_callback(True, 100, timer_callback)


showm.start()
