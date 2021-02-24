"""Module for testing primitive."""
import numpy as np
import numpy.testing as npt
from fury.utils import (map_coordinates_3d_4d,
                        vtk_matrix_to_numpy,
                        numpy_to_vtk_matrix,
                        get_grid_cells_position,
                        rotate, vtk, vertices_from_actor,
                        compute_bounds, set_input,
                        update_actor, get_actor_from_primitive,
                        get_bounds, pbr)
from fury import actor, window, utils
import fury.primitive as fp


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

    npt.assert_equal(utils.get_polydata_colors(vtk.vtkPolyData()), None)


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
    my_polydata = vtk.vtkPolyData()

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

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
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
    npt.assert_array_almost_equal(expected, res_vertices)


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
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    npt.assert_equal(compute_bounds(actor), None)
    npt.assert_equal(actor.GetMapper().GetInput().GetBounds(), test_bounds)


def test_update_actor():
    size = (15, 15)
    test_bounds = [0.0, 15,
                   0.0, 15,
                   0.0, 0.0]
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = vtk.vtkActor2D()
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
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(size[0], 0, 0)
    points.InsertNextPoint(size[0], size[1], 0)
    points.InsertNextPoint(0, size[1], 0)

    # Create the polygon
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    # Create a PolyData
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper2D()
    mapper = set_input(mapper, polygonPolyData)
    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    compute_bounds(actor)
    npt.assert_equal(get_bounds(actor), test_bounds)


def test_pbr(interactive=True):
    # Scene setup
    scene = window.Scene()

    """
    # NOTE: FAILS
    # Setup slicer
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine, value_range=[data.min(), data.max()])
    slicer.display(None, None, 25)
    slicer = pbr(slicer)
    scene.add(slicer)
    """

    # NOTE: Works
    # Setup surface
    import math
    import random
    from scipy.spatial import Delaunay
    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = - math.sin(i) * math.cos(j)
            fact2 = - math.exp(abs(1 - math.sqrt(i ** 2 + j ** 2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])
    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices
    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, "butterfly", "loop"]
    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                scene = window.Scene(background=(1, 1, 1))
                surface_actor = actor.surface(vertices, faces=face,
                                              colors=color, smooth=smooth_type)
                surface_actor = pbr(surface_actor)
                scene.add(surface_actor)

    """
    # NOTE: Works
    # Contour from roi setup
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.
    data[25, 20:30, 25] = 1.
    affine = np.eye(4)
    surface = actor.contour_from_roi(data, affine, color=np.array([1, 0, 1]))
    surface = pbr(surface)
    scene.add(surface)
    """

    """
    # NOTE: FAILS
    # Contour from label setup
    data = np.zeros((50, 50, 50))
    data[5:15, 1:10, 25] = 1.
    data[25:35, 1:10, 25] = 2.
    data[40:49, 1:10, 25] = 3.
    color = np.array([[255, 0, 0, 0.6],
                      [0, 255, 0, 0.5],
                      [0, 0, 255, 1.0]])
    surface = actor.contour_from_label(data, color=color)
    surface = pbr(surface)
    scene.add(surface)
    """

    """
    # NOTE: Works
    # Streamtube setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    tubes = actor.streamtube(data, colors, linewidth=.1)
    # TODO: Multiple metallic and roughness values
    tubes = pbr(tubes)
    scene.add(tubes)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # Line setup
    data1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    data2 = data1 + np.array([0.5, 0., 0.])
    data = [data1, data2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    lines = actor.line(data, colors, linewidth=5)
    lines = pbr(lines)
    scene.add(lines)
    """

    """
    # NOTE: FAILS
    # Scalar bar setup
    lut = actor.colormap_lookup_table(
        scale_range=(0., 100.), hue_range=(0., 0.1), saturation_range=(1, 1),
        value_range=(1., 1))
    sb_actor = actor.scalar_bar(lut, ' ')
    sb_actor = pbr(sb_actor)
    scene.add(sb_actor)
    """

    """
    # NOTE: Works
    # Axes setup
    axes = actor.axes()
    axes = pbr(axes)
    scene.add(axes)
    """

    """
    # NOTE: Works
    # ODF slicer setup
    from fury.optpkg import optional_package
    from tempfile import mkstemp
    dipy, have_dipy, _ = optional_package('dipy')
    if have_dipy:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric362')
        shape = (11, 11, 11, sphere.vertices.shape[0])
        fid, fname = mkstemp(suffix='_odf_slicer.mmap')
        odfs = np.memmap(fname, dtype=np.float64, mode='w+', shape=shape)
        odfs[:] = 1
        affine = np.eye(4)
        mask = np.ones(odfs.shape[:3])
        mask[:4, :4, :4] = 0
        odfs[..., 0] = 1
        odf_actor = actor.odf_slicer(odfs, affine, mask=mask, sphere=sphere,
                                     scale=.25, colormap='blues')
        odf_actor = pbr(odf_actor)
        scene.add(odf_actor)
    """

    """
    # NOTE: Works
    # Tensor slicer setup
    from fury.optpkg import optional_package
    dipy, have_dipy, _ = optional_package('dipy')
    if have_dipy:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        evals = np.array([1.4, .35, .35]) * 10 ** (-3)
        evecs = np.eye(3)
        mevals = np.zeros((3, 2, 4, 3))
        mevecs = np.zeros((3, 2, 4, 3, 3))
        mevals[..., :] = evals
        mevecs[..., :, :] = evecs
        affine = np.eye(4)
        scene = window.Scene()
        tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                           sphere=sphere, scale=.3)
        tensor_actor = pbr(tensor_actor)
        scene.add(tensor_actor)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # Peak slicer setup
    _peak_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f4')
    # peak_dirs.shape = (1, 1, 1) + peak_dirs.shape
    peak_dirs = np.zeros((11, 11, 11, 3, 3))
    peak_dirs[:, :, :] = _peak_dirs
    peak_actor = actor.peak_slicer(peak_dirs)
    peak_actor = pbr(peak_actor)
    scene.add(peak_actor)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # Dots setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    dots_actor = actor.dots(points, color=(0, 255, 0))
    dots_actor = pbr(dots_actor)
    scene.add(dots_actor)
    """

    """
    # NOTE: Works
    # Point setup
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    opacity = 0.5
    points_actor = actor.point(points, colors, opacity=opacity)
    points_actor = pbr(points_actor)
    scene.add(points_actor)
    """

    """
    # NOTE: Works
    # Sphere setup
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    opacity = 0.5
    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3], opacity=opacity)
    sphere_actor = pbr(sphere_actor)
    scene.add(sphere_actor)
    """

    """
    # NOTE: Works
    # Cylinder setup
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
    heights = np.array([5, 7, 10])
    cylinder_actor = actor.cylinder(xyz, dirs, colors, heights=heights)
    cylinder_actor = pbr(cylinder_actor)
    scene.add(cylinder_actor)
    """

    """
    # NOTE: Works
    # Square setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    square_actor = actor.square(centers, directions=directions, colors=colors,
                                scales=scale_list[3])
    square_actor = pbr(square_actor)
    scene.add(square_actor)
    """

    """
    # NOTE: Works. Same as square. Remove from final test and make comment
    # Rectangle setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    rectangle_actor = actor.rectangle(centers, directions=directions,
                                      colors=colors, scales=scale_list[3])
    rectangle_actor = pbr(rectangle_actor)
    scene.add(rectangle_actor)
    """

    """
    # NOTE: Works. Same as square. Remove from final test and make comment
    # Box setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    box_actor = actor.box(centers, directions=directions, colors=colors,
                          scales=scale_list[3])
    box_actor = pbr(box_actor)
    scene.add(box_actor)
    """

    """
    # NOTE: Works. Same as square. Remove from final test and make comment
    # Cube setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    cube_actor = actor.cube(centers, directions=directions, colors=colors,
                            scales=scale_list[3])
    cube_actor = pbr(cube_actor)
    scene.add(cube_actor)
    """

    """
    # NOTES: Works. Same as cylinder. Remove from final test and make comment
    # Arrow setup
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
    heights = np.array([5, 7, 10])
    arrow_actor = actor.arrow(xyz, dirs, colors, heights=heights)
    arrow_actor = pbr(arrow_actor)
    scene.add(arrow_actor)
    """

    """
    # NOTES: Works. Same as cylinder. Remove from final test and make comment
    # Cone setup
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
    heights = np.array([5, 7, 10])
    cone_actor = actor.cone(xyz, dirs, colors, heights=heights)
    cone_actor = pbr(cone_actor)
    scene.add(cone_actor)
    """

    """
    # NOTE: Works. Same as square. Remove from final test and make comment
    # Octagonalprism setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    octagonalprism_actor = actor.octagonalprism(centers, directions=directions,
                                                colors=colors,
                                                scales=scale_list[3])
    octagonalprism_actor = pbr(octagonalprism_actor)
    scene.add(octagonalprism_actor)
    """

    """
    # NOTE: Works. Same as square. Remove from final test and make comment
    # Frustum setup
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]
    frustum_actor = actor.frustum(centers, directions=directions,
                                  colors=colors, scales=scale_list[3])
    frustum_actor = pbr(frustum_actor)
    scene.add(frustum_actor)
    """

    """
    # NOTE: Works.
    # Superquadric setup
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.random.rand(3, 3)
    scales = [1, 2, 3]
    roundness = np.array([[1, 1], [1, 2], [2, 1]])

    sq_actor = actor.superquadric(centers, roundness=roundness,
                                  directions=directions,
                                  colors=colors.astype(np.uint8),
                                  scales=scales)
    sq_actor = pbr(sq_actor)
    scene.add(sq_actor)
    """

    """
    # NOTE: FAILS
    # Billboard setup
    centers = np.array([[0, 0, 0], [5, -5, 5], [-7, 7, -7], [10, 10, 10],
                        [10.5, 11.5, 11.5], [12, -12, -12], [-17, 17, 17],
                        [-22, -22, 22]])
    colors = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1],
                       [1, 0, 0], [0, 1, 0], [0, 1, 1]])
    scales = [6, .4, 1.2, 1, .2, .7, 3, 2]
    """
    fake_sphere = \
        """
        float len = length(point);
        float radius = 1.;
        if(len > radius)
            {discard;}

        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df_1 = max(0, dot(direction, normalizedPoint));
        float sf_1 = pow(df_1, 24);
        fragOutput0 = vec4(max(df_1 * color, sf_1 * vec3(1)), 1);
        """
    """
    billboard_actor = actor.billboard(centers, colors=colors, scales=scales,
                                      fs_impl=fake_sphere)
    billboard_actor = pbr(billboard_actor)
    scene.add(billboard_actor)
    """

    """
    # NOTE: Works
    # Label setup
    text_actor = actor.label("Hello")
    text_actor = pbr(text_actor)
    scene.add(text_actor)
    """

    """
    # NOTE: FAILS
    # Text3D setup
    msg = 'I \nlove\n FURY'
    txt_actor = actor.text_3d(msg)
    txt_actor = pbr(txt_actor)
    scene.add(txt_actor)
    """

    """
    # NOTE: FAILS
    # Figure setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, 3] = 0
    tp = actor.figure(arr)
    tp = pbr(tp)
    scene.add(tp)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # Texture setup
    arr = (255 * np.ones((512, 212, 4))).astype('uint8')
    arr[20:40, 20:40, :] = np.array([255, 0, 0, 255], dtype='uint8')
    tp2 = actor.texture(arr)
    tp2 = pbr(tp2)
    scene.add(tp2)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # Texture on sphere setup
    arr = 255 * np.ones((810, 1620, 3), dtype='uint8')
    rows, cols, _ = arr.shape
    rs = rows // 2
    cs = cols // 2
    w = 150 // 2
    arr[rs - w: rs + w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    tsa = actor.texture_on_sphere(arr)
    tsa = pbr(tsa)
    scene.add(tsa)
    """

    """
    # NOTE: Passes but doesn't seem to apply the effect
    # SDF setup
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]]) * 11
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    scales = [1, 2, 3]
    primitive = ['sphere', 'ellipsoid', 'torus']

    sdf_actor = actor.sdf(centers, directions=directions, colors=colors,
                          primitives=primitive, scales=scales)
    sdf_actor = pbr(sdf_actor)
    scene.add(sdf_actor)
    """

    if interactive:
        window.show(scene)
