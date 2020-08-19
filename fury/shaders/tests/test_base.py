from vtk.util import numpy_support
import numpy as np
import numpy.testing as npt

from fury import actor, window
from fury.shaders import (shader_to_actor, add_shader_callback,
                          attribute_to_actor, replace_shader_in_actor)


vertex_dec = \
    """
    uniform float time;
    out vec4 myVertexMC;
    mat4 rotationMatrix(vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;

        return mat4(oc * axis.x * axis.x + c,
                    oc * axis.x * axis.y - axis.z * s,
                    oc * axis.z * axis.x + axis.y * s,  0.0,
                    oc * axis.x * axis.y + axis.z * s,
                    oc * axis.y * axis.y + c,
                    oc * axis.y * axis.z - axis.x * s,  0.0,
                    oc * axis.z * axis.x - axis.y * s,
                    oc * axis.y * axis.z + axis.x * s,
                    oc * axis.z * axis.z + c,           0.0,
                    0.0, 0.0, 0.0, 1.0);
    }

    vec3 rotate(vec3 v, vec3 axis, float angle) {
        mat4 m = rotationMatrix(axis, angle);
        return (m * vec4(v, 1.0)).xyz;
    }

    vec3 ax = vec3(1, 0, 0);
    """

vertex_impl = \
    """
    myVertexMC = vertexMC;
    myVertexMC.xyz = rotate(vertexMC.xyz, ax, time*0.01);
        vertexVCVSOutput = MCVCMatrix * myVertexMC;
        gl_Position = MCDCMatrix * myVertexMC;
    """

frag_dec = \
    """
    varying vec4 myVertexMC;
    uniform float time;
    """

frag_impl = \
    """
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
    """


def generate_cube_with_effect():
    cube = actor.cube(np.array([[0, 0, 0]]))
    shader_to_actor(cube, "vertex", impl_code=vertex_impl,
                    decl_code=vertex_dec, block="valuepass")
    shader_to_actor(cube, "fragment", impl_code=frag_impl,
                    decl_code=frag_dec, block="light")
    return cube


def test_shader_to_actor(interactive=False):
    cube = generate_cube_with_effect()

    scene = window.Scene()
    scene.add(cube)
    if interactive:
        scene.add(actor.axes())
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    # test errors
    npt.assert_raises(ValueError, shader_to_actor, cube, "error",
                      vertex_impl)
    npt.assert_raises(ValueError, shader_to_actor, cube, "vertex",
                      vertex_impl, block="error")
    npt.assert_raises(ValueError, replace_shader_in_actor, cube, "error",
                      vertex_impl)


def test_add_shader_callback():
    cube = generate_cube_with_effect()
    showm = window.ShowManager()
    showm.scene.add(cube)
    timer = 0

    def timer_callback(obj, event):
        nonlocal timer, showm
        timer += 1.0
        showm.render()
        if timer > 9:
            showm.exit()

    def my_cbk(_caller, _event, calldata=None):
        program = calldata
        nonlocal timer
        if program is not None:
            try:
                program.SetUniformf("time", timer)
            except ValueError:
                pass

    add_shader_callback(cube, my_cbk)
    showm.initialize()
    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()

    arr = window.snapshot(showm.scene, offscreen=True)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)


def test_attribute_to_actor():
    cube = generate_cube_with_effect()
    test_arr = np.arange(24).reshape((8, 3))

    attribute_to_actor(cube, test_arr, 'test_arr')

    arr = cube.GetMapper().GetInput().GetPointData().GetArray('test_arr')
    npt.assert_array_equal(test_arr, numpy_support.vtk_to_numpy(arr))
