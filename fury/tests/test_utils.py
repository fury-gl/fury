"""Module for testing primitive."""
import pytest
import numpy as np
import numpy.testing as npt
from fury.utils import (add_polydata_numeric_field, get_polydata_field,
                        get_polydata_tangents, map_coordinates_3d_4d,
                        normals_from_actor, normals_to_actor,
                        set_polydata_tangents, tangents_from_actor,
                        tangents_from_direction_of_anisotropy,
                        tangents_to_actor, vtk_matrix_to_numpy,
                        numpy_to_vtk_matrix,
                        get_grid_cells_position,
                        rotate, vertices_from_actor,
                        compute_bounds, set_input,
                        update_actor, get_actor_from_primitive,
                        get_bounds, update_surface_actor_colors,
                        apply_affine_to_actor)
from fury import actor, window, utils
from fury.lib import (numpy_support, PolyData, PolyDataMapper2D, Points,
                      CellArray, Polygon, Actor2D, DoubleArray,
                      UnsignedCharArray, TextActor3D, VTK_DOUBLE, VTK_FLOAT)
import fury.primitive as fp


def test_apply_affine_to_actor(interactive=False):
    text_act = actor.text_3d("ALIGN TOP RIGHT", justification='right',
                             vertical_justification='top')

    text_act2 = TextActor3D()
    text_act2.SetInput("ALIGN TOP RIGHT")
    text_act2.GetTextProperty().SetFontFamilyToArial()
    text_act2.GetTextProperty().SetFontSize(24)
    text_act2.SetScale((1./24.*12,)*3)

    if interactive:
        scene = window.Scene()
        scene.add(text_act, text_act2)
        window.show(scene)

    text_bounds = [0, 0, 0, 0]
    text_act2.GetBoundingBox(text_bounds)
    initial_bounds = text_act2.GetBounds()

    affine = np.eye(4)
    affine[:3, -1] += (-text_bounds[1], 0, 0)
    affine[:3, -1] += (0, -text_bounds[3], 0)
    affine[:3, -1] *= text_act2.GetScale()
    apply_affine_to_actor(text_act2, affine)
    text_act2.GetBoundingBox(text_bounds)

    if interactive:
        scene = window.Scene()
        scene.add(text_act, text_act2)
        window.show(scene)

    updated_bounds = text_act2.GetBounds()
    original_bounds = text_act.GetBounds()
    npt.assert_array_almost_equal(updated_bounds, original_bounds, decimal=0)

    def compare(x, y):
        return np.isclose(x, y, rtol=1)
    npt.assert_array_compare(compare, updated_bounds, original_bounds)


def test_map_coordinates_3d_4d():
    data_1 = np.zeros((5, 5, 5))
    data_1[2, 2, 2] = 1
    data_2 = np.zeros((5, 5, 5, 5))
    data_2[2, 2, 2] = 1

    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1.5, 1.5, 1.5]])
    expected = np.array([0, 0, 1, 0.125])
    expected2 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [0.125, 0.125, 0.125, 0.125, 0.125]])

    for d, e in zip([data_1, data_2], [expected, expected2]):
        values = map_coordinates_3d_4d(d, indices)
        npt.assert_array_almost_equal(values, e)

    # Test error
    npt.assert_raises(ValueError, map_coordinates_3d_4d, np.ones(5), indices)
    npt.assert_raises(ValueError, map_coordinates_3d_4d,
                      np.ones((5, 5, 5, 5, 5)), indices)


def test_polydata_lines():
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    line_1 = np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3.]])
    line_2 = line_1 + np.array([0.5, 0., 0.])
    lines = [line_1, line_2]

    pd_lines, is_cmap = utils.lines_to_vtk_polydata(lines, colors)
    res_lines = utils.get_polydata_lines(pd_lines)
    npt.assert_array_equal(lines, res_lines)
    npt.assert_equal(is_cmap, False)

    res_colors = utils.get_polydata_colors(pd_lines)
    res_colors = np.unique(res_colors, axis=0) / 255
    npt.assert_array_equal(colors, np.flipud(res_colors))

    npt.assert_equal(utils.get_polydata_colors(PolyData()), None)


def test_polydata_polygon(interactive=False):
    # Create a cube
    my_triangles = np.array([[0, 6, 4],
                             [0, 2, 6],
                             [0, 3, 2],
                             [0, 1, 3],
                             [2, 7, 6],
                             [2, 3, 7],
                             [4, 6, 7],
                             [4, 7, 5],
                             [0, 4, 5],
                             [0, 5, 1],
                             [1, 5, 7],
                             [1, 7, 3]], dtype='i8')
    my_vertices = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0]])
    colors = my_vertices * 255
    my_polydata = PolyData()

    utils.set_polydata_vertices(my_polydata, my_vertices)
    utils.set_polydata_triangles(my_polydata, my_triangles)

    npt.assert_equal(len(my_vertices), my_polydata.GetNumberOfPoints())
    npt.assert_equal(len(my_triangles), my_polydata.GetNumberOfCells())
    npt.assert_equal(utils.get_polydata_normals(my_polydata), None)

    res_triangles = utils.get_polydata_triangles(my_polydata)
    res_vertices = utils.get_polydata_vertices(my_polydata)

    npt.assert_array_equal(my_vertices, res_vertices)
    npt.assert_array_equal(my_triangles, res_triangles)

    utils.set_polydata_colors(my_polydata, colors)
    npt.assert_equal(utils.get_polydata_colors(my_polydata), colors)

    utils.update_polydata_normals(my_polydata)
    normals = utils.get_polydata_normals(my_polydata)
    npt.assert_equal(len(normals), len(my_vertices))

    mapper = utils.get_polymapper_from_polydata(my_polydata)
    actor1 = utils.get_actor_from_polymapper(mapper)
    actor2 = utils.get_actor_from_polydata(my_polydata)

    scene = window.Scene()
    for act in [actor1, actor2]:
        scene.add(act)
        if interactive:
            window.show(scene)
        arr = window.snapshot(scene)

        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 1)


def test_add_polydata_numeric_field():
    my_polydata = PolyData()
    poly_field_data = my_polydata.GetFieldData()
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    bool_data = True
    add_polydata_numeric_field(my_polydata, 'Test Bool', bool_data)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_equal(poly_field_data.GetArray('Test Bool').GetValue(0),
                     bool_data)
    poly_field_data.RemoveArray('Test Bool')
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    int_data = 1
    add_polydata_numeric_field(my_polydata, 'Test Int', int_data)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_equal(poly_field_data.GetArray('Test Int').GetValue(0),
                     int_data)
    poly_field_data.RemoveArray('Test Int')
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    float_data = .1
    add_polydata_numeric_field(my_polydata, 'Test Float', float_data,
                               array_type=VTK_FLOAT)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_almost_equal(poly_field_data.GetArray('Test Float').GetValue(0),
                            float_data)
    poly_field_data.RemoveArray('Test Float')
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    double_data = .1
    add_polydata_numeric_field(my_polydata, 'Test Double', double_data,
                               array_type=VTK_DOUBLE)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_equal(poly_field_data.GetArray('Test Double').GetValue(0),
                     double_data)
    poly_field_data.RemoveArray('Test Double')
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    array_data = [-1, 0, 1]
    add_polydata_numeric_field(my_polydata, 'Test Array', array_data)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_equal(
        numpy_support.vtk_to_numpy(poly_field_data.GetArray('Test Array')),
        array_data)
    poly_field_data.RemoveArray('Test Array')
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 0)
    ndarray_data = np.array([[-.1, -.1], [0, 0], [.1, .1]])
    add_polydata_numeric_field(my_polydata, 'Test NDArray', ndarray_data,
                               array_type=VTK_FLOAT)
    npt.assert_equal(poly_field_data.GetNumberOfArrays(), 1)
    npt.assert_almost_equal(
        numpy_support.vtk_to_numpy(poly_field_data.GetArray('Test NDArray')),
        ndarray_data)


def test_get_polydata_field():
    my_polydata = PolyData()
    field_data = get_polydata_field(my_polydata, 'Test')
    npt.assert_equal(field_data, None)
    data = 1
    field_name = 'Test'
    vtk_data = numpy_support.numpy_to_vtk(data)
    vtk_data.SetName(field_name)
    my_polydata.GetFieldData().AddArray(vtk_data)
    field_data = get_polydata_field(my_polydata, field_name)
    npt.assert_equal(field_data, data)


def test_get_polydata_tangents():
    my_polydata = PolyData()
    tangents = get_polydata_tangents(my_polydata)
    npt.assert_equal(tangents, None)
    array = np.array([[0, 0, 0], [1, 1, 1]])
    my_polydata.GetPointData().SetTangents(
        numpy_support.numpy_to_vtk(array, deep=True, array_type=VTK_FLOAT))
    tangents = get_polydata_tangents(my_polydata)
    npt.assert_array_equal(tangents, array)


def test_set_polydata_tangents():
    my_polydata = PolyData()
    poly_point_data = my_polydata.GetPointData()
    npt.assert_equal(poly_point_data.GetNumberOfArrays(), 0)
    array = np.array([[0, 0, 0], [1, 1, 1]])
    set_polydata_tangents(my_polydata, array)
    npt.assert_equal(poly_point_data.GetNumberOfArrays(), 1)
    npt.assert_equal(poly_point_data.HasArray('Tangents'), True)


def test_asbytes():
    text = [b'test', 'test']

    for t in text:
        npt.assert_equal(utils.asbytes(t), b'test')


def trilinear_interp_numpy(input_array, indices):
    """Evaluate the input_array data at the given indices."""
    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    x_indices = indices[:, 0]
    y_indices = indices[:, 1]
    z_indices = indices[:, 2]

    x0 = x_indices.astype(int)
    y0 = y_indices.astype(int)
    z0 = z_indices.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Check if xyz1 is beyond array boundary:
    x1[np.where(x1 == input_array.shape[0])] = x0.max()
    y1[np.where(y1 == input_array.shape[1])] = y0.max()
    z1[np.where(z1 == input_array.shape[2])] = z0.max()

    if input_array.ndim == 3:
        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0

    elif input_array.ndim == 4:
        x = np.expand_dims(x_indices - x0, axis=1)
        y = np.expand_dims(y_indices - y0, axis=1)
        z = np.expand_dims(z_indices - z0, axis=1)

    output = (input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) +
              input_array[x1, y0, z0] * x * (1 - y) * (1 - z) +
              input_array[x0, y1, z0] * (1 - x) * y * (1-z) +
              input_array[x0, y0, z1] * (1 - x) * (1 - y) * z +
              input_array[x1, y0, z1] * x * (1 - y) * z +
              input_array[x0, y1, z1] * (1 - x) * y * z +
              input_array[x1, y1, z0] * x * y * (1 - z) +
              input_array[x1, y1, z1] * x * y * z)

    return output


def test_trilinear_interp():

    A = np.zeros((5, 5, 5))
    A[2, 2, 2] = 1

    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1.5, 1.5, 1.5]])

    values = trilinear_interp_numpy(A, indices)
    values2 = map_coordinates_3d_4d(A, indices)
    npt.assert_almost_equal(values, values2)

    B = np.zeros((5, 5, 5, 3))
    B[2, 2, 2] = np.array([1, 1, 1])

    values = trilinear_interp_numpy(B, indices)
    values_4d = map_coordinates_3d_4d(B, indices)
    npt.assert_almost_equal(values, values_4d)


def test_vtk_matrix_to_numpy():

    A = np.array([[2., 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 1]])

    for shape in [3, 4]:
        vtkA = numpy_to_vtk_matrix(A[:shape, :shape])
        Anew = vtk_matrix_to_numpy(vtkA)
        npt.assert_array_almost_equal(A[:shape, :shape], Anew)

    npt.assert_equal(vtk_matrix_to_numpy(None), None)
    npt.assert_equal(numpy_to_vtk_matrix(None), None)
    npt.assert_raises(ValueError, numpy_to_vtk_matrix, np.array([A, A]))


def test_numpy_to_vtk_image_data():
    array = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[21, 22, 23],
                       [24, 25, 26]],
                      [[27, 28, 29],
                       [210, 211, 212]]])

    # converting numpy array to vtk_image_data
    vtk_image_data = utils.numpy_to_vtk_image_data(array)

    # extracting the image data from vtk_image_data
    w, h, depth = vtk_image_data.GetDimensions()
    vtk_img_array = vtk_image_data.GetPointData().GetScalars()
    elements = vtk_img_array.GetNumberOfComponents()

    # converting vtk array to numpy array
    numpy_img_array = numpy_support.vtk_to_numpy(vtk_img_array)
    npt.assert_equal(np.flipud(array), numpy_img_array.reshape(h, w, elements))

    npt.assert_raises(IOError, utils.numpy_to_vtk_image_data,
                      np.array([1, 2, 3]))

def test_get_grid_cell_position():

    shapes = 10 * [(50, 50), (50, 50), (50, 50), (80, 50)]

    npt.assert_raises(ValueError, get_grid_cells_position, shapes=shapes,
                      dim=(1, 1))

    CS = get_grid_cells_position(shapes=shapes)
    npt.assert_equal(CS.shape, (42, 3))
    npt.assert_almost_equal(CS[-1], [480., -250., 0])


def test_rotate(interactive=False):

    A = np.zeros((50, 50, 50))

    A[20:30, 20:30, 10:40] = 100

    act = actor.contour_from_roi(A)

    scene = window.Scene()

    scene.add(act)

    if interactive:
        window.show(scene)
    else:
        arr = window.snapshot(scene, offscreen=True)
        red = arr[..., 0].sum()
        red_sum = np.sum(red)

    act2 = utils.shallow_copy(act)

    rot = (90, 1, 0, 0)

    rotate(act2, rot)

    act3 = utils.shallow_copy(act)

    scene.add(act2)

    rot = (90, 0, 1, 0)

    rotate(act3, rot)

    scene.add(act3)

    scene.add(actor.axes())

    if interactive:
        window.show(scene)
    else:

        arr = window.snapshot(scene, offscreen=True)
        red_sum_new = arr[..., 0].sum()
        npt.assert_equal(red_sum_new > red_sum, True)


def test_triangle_order():

    test_vert = np.array([[-1, -2, 0],
                          [1, -1, 0],
                          [2, 1, 0],
                          [3, 0, 0]])

    test_tri = np.array([[0, 1, 2],
                         [2, 1, 0]])

    clockwise1 = utils.triangle_order(test_vert, test_tri[0])
    clockwise2 = utils.triangle_order(test_vert, test_tri[1])

    npt.assert_equal(False, clockwise1)
    npt.assert_equal(False, clockwise2)


def test_change_vertices_order():

    triangles = np.array([[1, 2, 3],
                          [3, 2, 1],
                          [5, 4, 3],
                          [3, 4, 5]])

    npt.assert_equal(triangles[0], utils.change_vertices_order(triangles[1]))
    npt.assert_equal(triangles[2], utils.change_vertices_order(triangles[3]))


def test_winding_order():

    vertices = np.array([[0, 0, 0],
                         [1, 2, 0],
                         [3, 0, 0],
                         [2, 0, 0]])

    triangles = np.array([[0, 1, 3],
                          [2, 1, 0]])

    expected_triangles = np.array([[0, 1, 3],
                                   [2, 1, 0]])

    npt.assert_equal(expected_triangles,
                     utils.fix_winding_order(vertices, triangles))


def test_vertices_from_actor(interactive=False):

    expected = np.array([[1.5, -0.5, 0.],
                         [1.5, 0.5, 0],
                         [2.5, 0.5, 0],
                         [2.5, -0.5, 0],
                         [-1, 1, 0],
                         [-1, 3, 0],
                         [1, 3, 0],
                         [1, 1, 0],
                         [-0.5, -0.5, 0],
                         [-0.5, 0.5, 0],
                         [0.5, 0.5, 0],
                         [0.5, -0.5, 0]])
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    scales = [1, 2, 1]
    verts, faces = fp.prim_square()
    res = fp.repeat_primitive(verts, faces, centers=centers, colors=colors,
                              scales=scales)

    big_verts = res[0]
    big_faces = res[1]
    big_colors = res[2]
    actr = get_actor_from_primitive(big_verts, big_faces, big_colors)
    actr.GetProperty().BackfaceCullingOff()
    if interactive:
        scene = window.Scene()
        scene.add(actor.axes())
        scene.add(actr)
        window.show(scene)
    res_vertices = vertices_from_actor(actr)
    res_vertices_vtk = vertices_from_actor(actr, as_vtk=True)

    npt.assert_array_almost_equal(expected, res_vertices)
    npt.assert_equal(isinstance(res_vertices_vtk, DoubleArray), True)

    # test colors_from_actor:
    l_colors = utils.colors_from_actor(actr)
    l_colors_vtk = utils.colors_from_actor(actr, as_vtk=True)
    l_colors_none = utils.colors_from_actor(actr, array_name='col')

    npt.assert_equal(l_colors_none, None)
    npt.assert_equal(isinstance(l_colors_vtk, UnsignedCharArray), True)
    npt.assert_equal(np.unique(l_colors, axis=0).shape, colors.shape)

    l_array = utils.array_from_actor(actr, 'colors')
    l_array_vtk = utils.array_from_actor(actr, 'colors', as_vtk=True)
    l_array_none = utils.array_from_actor(actr, 'col')

    npt.assert_array_equal(l_array, l_colors)
    npt.assert_equal(l_array_none, None)
    npt.assert_equal(isinstance(l_array_vtk, UnsignedCharArray), True)


def test_normals_from_actor():
    my_actor = actor.square(np.array([[0, 0, 0]]))
    normals = normals_from_actor(my_actor)
    npt.assert_equal(normals, None)
    array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    my_actor.GetMapper().GetInput().GetPointData().SetNormals(
        numpy_support.numpy_to_vtk(array, deep=True))
    normals = normals_from_actor(my_actor)
    npt.assert_array_equal(normals, array)


def test_normals_to_actor():
    my_actor = actor.square(np.array([[0, 0, 0]]))
    poly_point_data = my_actor.GetMapper().GetInput().GetPointData()
    npt.assert_equal(poly_point_data.HasArray('Normals'), False)
    array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    normals_to_actor(my_actor, array)
    npt.assert_equal(poly_point_data.HasArray('Normals'), True)
    normals = numpy_support.vtk_to_numpy(poly_point_data.GetArray('Normals'))
    npt.assert_array_equal(normals, array)


def test_tangents_from_actor():
    my_actor = actor.square(np.array([[0, 0, 0]]))
    tangents = tangents_from_actor(my_actor)
    npt.assert_equal(tangents, None)
    array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    my_actor.GetMapper().GetInput().GetPointData().SetTangents(
        numpy_support.numpy_to_vtk(array, deep=True, array_type=VTK_FLOAT))
    tangents = tangents_from_actor(my_actor)
    npt.assert_array_equal(tangents, array)


def test_tangents_from_direction_of_anisotropy():
    normals = np.array([[-1., 0., 0.], [0., 0., 1.]])
    doa = (0., 1., 0.)
    expected = np.array([[0., 0., 1.], [1., 0., 0.]])
    actual = tangents_from_direction_of_anisotropy(normals, doa)
    npt.assert_array_equal(actual, expected)


def test_tangents_to_actor():
    my_actor = actor.square(np.array([[0, 0, 0]]))
    poly_point_data = my_actor.GetMapper().GetInput().GetPointData()
    npt.assert_equal(poly_point_data.HasArray('Tangents'), False)
    array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    tangents_to_actor(my_actor, array)
    npt.assert_equal(poly_point_data.HasArray('Tangents'), True)
    tangents = numpy_support.vtk_to_numpy(poly_point_data.GetArray('Tangents'))
    npt.assert_array_equal(tangents, array)


def test_get_actor_from_primitive():
    vertices, triangles = fp.prim_frustum()
    colors = np.array([1, 0, 0])
    npt.assert_raises(ValueError, get_actor_from_primitive, vertices,
                      triangles, colors=colors)


def test_compute_bounds():
    size = (15, 15)
    test_bounds = [0.0, 15,
                   0.0, 15,
                   0.0, 0.0]
    points = Points()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = Polygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = CellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = PolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = PolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = Actor2D()
    actor.SetMapper(mapper)
    npt.assert_equal(compute_bounds(actor), None)
    npt.assert_equal(actor.GetMapper().GetInput().GetBounds(), test_bounds)


def test_update_actor():
    size = (15, 15)
    test_bounds = [0.0, 15,
                   0.0, 15,
                   0.0, 0.0]
    points = Points()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = Polygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = CellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = PolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = PolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = Actor2D()
    actor.SetMapper(mapper)
    compute_bounds(actor)
    npt.assert_equal(actor.GetMapper().GetInput().GetBounds(), test_bounds)
    updated_size = (35, 35)
    points.SetPoint(0, 0, 0, 0.0)
    points.SetPoint(1, updated_size[0], 0, 0.0)
    points.SetPoint(2, updated_size[0], updated_size[1], 0.0)
    points.SetPoint(3, 0, updated_size[1], 0.0)
    polygonPolyData.SetPoints(points)
    test_bounds = [0.0, 35.0,
                   0.0, 35.0,
                   0.0, 0.0]
    compute_bounds(actor)
    npt.assert_equal(None, update_actor(actor))
    npt.assert_equal(test_bounds, actor.GetMapper().GetInput().GetBounds())


def test_get_bounds():
    size = (15, 15)
    test_bounds = [0.0, 15,
                   0.0, 15,
                   0.0, 0.0]
    points = Points()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = Polygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = CellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = PolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = PolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = Actor2D()
    actor.SetMapper(mapper)
    compute_bounds(actor)
    npt.assert_equal(get_bounds(actor), test_bounds)


def test_update_surface_actor_colors():
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    x, y = np.meshgrid(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = x**2 + y**2
    colors = np.array([[0.2, 0.4, 0.8]]*400)
    xyz = np.vstack([x, y, z]).T
    act = actor.surface(xyz)
    update_surface_actor_colors(act, colors)

    # Multiplying colors by 255 to convert them into RGB format used by VTK.
    colors *= 255

    # colors obtained from the surface
    surface_colors = numpy_support.vtk_to_numpy(act.GetMapper().GetInput().
                                                GetPointData().GetScalars())

    # Checking if the colors passed to the function and colors assigned are
    # same.
    npt.assert_equal(colors, surface_colors)
