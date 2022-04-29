import os
import pytest

import numpy as np
import numpy.testing as npt

from fury import actor, window
from fury.shaders import (add_shader_callback, attribute_to_actor,
                          compose_shader, import_fury_shader, load_shader,
                          load, shader_to_actor, replace_shader_in_actor)
from fury.shaders.base import SHADERS_DIR
from fury.lib import (Actor, CellArray, Points, PolyData, PolyDataMapper,
                      numpy_support)
from fury.utils import set_polydata_colors
from tempfile import TemporaryDirectory as InTemporaryDirectory


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

geometry_code = \
    """
    //VTK::System::Dec
    //VTK::PositionVC::Dec
    uniform mat4 MCDCMatrix;

    //VTK::PrimID::Dec

    // declarations below aren't necessary because they are already injected
    // by PrimID template this comment is just to justify the passthrough below
    //in vec4 vertexColorVSOutput[];
    //out vec4 vertexColorGSOutput;

    //VTK::Color::Dec
    //VTK::Normal::Dec
    //VTK::Light::Dec
    //VTK::TCoord::Dec
    //VTK::Picking::Dec
    //VTK::DepthPeeling::Dec
    //VTK::Clip::Dec
    //VTK::Output::Dec

    // Convert points to line strips
    layout(points) in;
    layout(triangle_strip, max_vertices = 4) out;

    void build_square(vec4 position)
    {
        gl_Position = position + vec4(-.5, -.5, 0, 0);  // 1: Bottom left
        EmitVertex();
        gl_Position = position + vec4(.5, -.5, 0, 0);  // 2: Bottom right
        EmitVertex();
        gl_Position = position + vec4(-.5, .5, 0, 0);  // 3: Top left
        EmitVertex();
        gl_Position = position + vec4(.5, .5, 0, 0);  // 4: Top right
        EmitVertex();
        EndPrimitive();
    }

    void main()
    {
    vertexColorGSOutput = vertexColorVSOutput[0];
    build_square(gl_in[0].gl_Position);
    }
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


def generate_points():
    centers = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255

    vtk_vertices = Points()
    # Create the topology of the point (a vertex)
    vtk_faces = CellArray()
    # Add points
    for i in range(len(centers)):
        p = centers[i]
        id = vtk_vertices.InsertNextPoint(p)
        vtk_faces.InsertNextCell(1)
        vtk_faces.InsertCellPoint(id)
    # Create a polydata object
    polydata = PolyData()
    # Set the vertices and faces we created as the geometry and topology of the
    # polydata
    polydata.SetPoints(vtk_vertices)
    polydata.SetVerts(vtk_faces)

    set_polydata_colors(polydata, colors)

    mapper = PolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetVBOShiftScaleMethod(False)

    point_actor = Actor()
    point_actor.SetMapper(mapper)

    return point_actor


def test_add_shader_callback():
    cube = generate_cube_with_effect()
    showm = window.ShowManager()
    showm.scene.add(cube)
    class Timer(object):
        idx = 0.0

    timer = Timer()

    def timer_callback(obj, event):
        # nonlocal timer, showm
        timer.idx += 1.0
        showm.render()
        if timer.idx > 90:
            showm.exit()

    def my_cbk(_caller, _event, calldata=None):
        program = calldata

        if program is not None:
            try:
                program.SetUniformf("time", timer.idx)
            except ValueError:
                pass

    add_shader_callback(cube, my_cbk)
    showm.initialize()
    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()

    arr = window.snapshot(showm.scene, offscreen=True)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 1)

    cone_actor = actor.cone(np.array([[0, 0, 0]]), np.array([[0, 1, 0]]),
                            (0, 0, 1))

    test_values = []

    def callbackLow(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(0)

    id_observer = add_shader_callback(cone_actor, callbackLow, 0)

    with pytest.raises(Exception):
        add_shader_callback(cone_actor, callbackLow, priority='str')

    mapper = cone_actor.GetMapper()
    mapper.RemoveObserver(id_observer)

    scene = window.Scene()
    scene.add(cone_actor)

    arr1 = window.snapshot(scene, size=(200, 200))
    assert len(test_values) == 0

    test_values = []

    def callbackHigh(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(999)

    def callbackMean(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(500)

    add_shader_callback(cone_actor, callbackHigh, 999)
    add_shader_callback(cone_actor, callbackLow, 0)

    id_mean = add_shader_callback(cone_actor, callbackMean, 500)

    # check the priority of each call
    arr2 = window.snapshot(scene, size=(200, 200))
    assert np.abs([
        test_values[0] - 999, test_values[1] - 500,
        test_values[2] - 0]).sum() == 0

    # check if the correct observer was removed
    mapper.RemoveObserver(id_mean)
    test_values = []

    arr3 = window.snapshot(scene, size=(200, 200))
    assert np.abs([
        test_values[0] - 999, test_values[1] - 0]).sum() == 0


def test_attribute_to_actor():
    cube = generate_cube_with_effect()
    test_arr = np.arange(24).reshape((8, 3))

    attribute_to_actor(cube, test_arr, 'test_arr')

    arr = cube.GetMapper().GetInput().GetPointData().GetArray('test_arr')
    npt.assert_array_equal(test_arr, numpy_support.vtk_to_numpy(arr))


def test_compose_shader():
    str_test1 = 'Test1'
    str_test2 = 'Test2'
    list_str1 = [str_test1, None]
    list_str2 = [str_test1, str_test2]
    # Test empty parameter
    code = compose_shader(None)
    npt.assert_equal(code, "")
    # Test invalid list
    npt.assert_raises(IOError, compose_shader, list_str1)
    # Test str code
    code = compose_shader(str_test1)
    npt.assert_equal(code, str_test1)
    # Test list of str code
    code = compose_shader(list_str2)
    npt.assert_equal(code, '\n' + '\n'.join(list_str2))


def test_import_fury_shader():
    str_test1 = 'Test1'
    fname_test1 = 'test1.frag'
    fname_test2 = 'test2.txt'
    pname_test1 = os.path.join(SHADERS_DIR, fname_test1)

    # Test invalid file extension
    npt.assert_raises(IOError, import_fury_shader, fname_test2)

    # Test file not found
    npt.assert_raises(IOError, import_fury_shader, pname_test1)

    # Test valid file
    with open(pname_test1, 'w') as f:
        f.write(str_test1)
    code = import_fury_shader(fname_test1)
    npt.assert_equal(code, str_test1)

    os.remove(pname_test1)


def test_load_shader():
    fname_test = 'test.text'

    # Test invalid file extension
    npt.assert_raises(IOError, load_shader, fname_test)

    with InTemporaryDirectory() as tdir:
        fname_test = 'test.frag'
        fname_test = os.path.join(tdir, fname_test)
        str_test = 'Test1'
        test_file = open(fname_test, 'w')
        test_file.write(str_test)
        test_file.close()

        npt.assert_string_equal(load_shader(fname_test), str_test)


def test_load():
    dummy_file_name = 'dummy.txt'
    dummy_file_contents = 'This is some dummy text.'

    dummy_file = open(os.path.join(SHADERS_DIR, dummy_file_name), 'w')
    dummy_file.write(dummy_file_contents)
    dummy_file.close()

    npt.assert_warns(DeprecationWarning, load, dummy_file_name)

    npt.assert_string_equal(load(dummy_file_name), dummy_file_contents)

    os.remove(os.path.join(SHADERS_DIR, dummy_file_name))


def test_replace_shader_in_actor(interactive=False):
    scene = window.Scene()
    test_actor = generate_points()
    scene.add(test_actor)
    if interactive:
        window.show(scene)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[40, 140, :]
    npt.assert_array_equal(actual, [0, 0, 0])
    actual = ss[140, 40, :]
    npt.assert_array_equal(actual, [0, 0, 0])
    actual = ss[40, 40, :]
    npt.assert_array_equal(actual, [0, 0, 0])
    scene.clear()
    replace_shader_in_actor(test_actor, 'geometry', geometry_code)
    scene.add(test_actor)
    if interactive:
        window.show(scene)
    ss = window.snapshot(scene, size=(200, 200))
    actual = ss[40, 140, :]
    npt.assert_array_equal(actual, [255, 0, 0])
    actual = ss[140, 40, :]
    npt.assert_array_equal(actual, [0, 255, 0])
    actual = ss[40, 40, :]
    npt.assert_array_equal(actual, [0, 0, 255])


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
    npt.assert_raises(ValueError, shader_to_actor, cube, "geometry",
                      vertex_impl)
    npt.assert_raises(ValueError, shader_to_actor, cube, "vertex",
                      vertex_impl, block="error")
    npt.assert_raises(ValueError, replace_shader_in_actor, cube, "error",
                      vertex_impl)
