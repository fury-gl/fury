import numpy as np
from scipy.spatial import Delaunay
import vtk
from vtk.util import numpy_support

from fury.utils import (
    get_actor_from_polydata,
    numpy_to_vtk_colors,
    numpy_to_vtk_points,
    set_polydata_triangles,
    set_polydata_vertices,
)


def cube():
    my_polydata = vtk.vtkPolyData()

    my_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    my_vertices -= 0.5

    my_triangles = np.array(
        [
            [0, 6, 4],
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
            [1, 7, 3],
        ],
        dtype='i8',
    )

    set_polydata_vertices(my_polydata, my_vertices)
    set_polydata_triangles(my_polydata, my_triangles)
    return get_actor_from_polydata(my_polydata)


def disk():

    np.random.seed(42)
    n_points = 1
    centers = np.random.rand(n_points, 3)
    colors = 255 * np.random.rand(n_points, 3)

    vtk_points = numpy_to_vtk_points(centers)

    points_polydata = vtk.vtkPolyData()
    points_polydata.SetPoints(vtk_points)

    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(points_polydata)
    vertex_filter.Update()

    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(vertex_filter.GetOutput())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    points_actor = vtk.vtkActor()
    points_actor.SetMapper(mapper)
    points_actor.GetProperty().SetPointSize(1000)
    points_actor.GetProperty().SetRenderPointsAsSpheres(True)

    return points_actor


def glyph_dot(centers, colors, radius=100):
    if np.array(colors).ndim == 1:
        colors = np.tile(colors, (len(centers), 1))

    vtk_pnt = numpy_to_vtk_points(centers)

    pnt_polydata = vtk.vtkPolyData()
    pnt_polydata.SetPoints(vtk_pnt)

    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(pnt_polydata)
    vertex_filter.Update()

    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(vertex_filter.GetOutput())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUsePointFieldData()

    pnt_actor = vtk.vtkActor()
    pnt_actor.SetMapper(mapper)

    pnt_actor.GetProperty().SetPointSize(radius)

    cols = numpy_to_vtk_colors(255 * np.ascontiguousarray(colors))
    cols.SetName('colors')
    polydata.GetPointData().AddArray(cols)
    mapper.SelectColorArray('colors')

    return pnt_actor


def rectangle(size=(1, 1)):
    X, Y = size

    # Setup four points
    points = vtk.vtkPoints()
    points.InsertNextPoint(-X / 2, -Y / 2, 0)
    points.InsertNextPoint(-X / 2, Y / 2, 0)
    points.InsertNextPoint(X / 2, Y / 2, 0)
    points.InsertNextPoint(X / 2, -Y / 2, 0)

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
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def rectangle2(centers, colors, use_vertices=False, size=(2, 2)):
    """Visualize one or many spheres with different colors and radii

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]

    """
    if np.array(colors).ndim == 1:
        colors = np.tile(colors, (len(centers), 1))

    pts = numpy_to_vtk_points(np.ascontiguousarray(centers))
    cols = numpy_to_vtk_colors(255 * np.ascontiguousarray(colors))
    cols.SetName('colors')

    polydata_centers = vtk.vtkPolyData()

    polydata_centers.SetPoints(pts)
    polydata_centers.GetPointData().AddArray(cols)

    print('NB pts: ', polydata_centers.GetNumberOfPoints())
    print('NB arrays: ', polydata_centers.GetPointData().GetNumberOfArrays())

    for i in range(polydata_centers.GetPointData().GetNumberOfArrays()):
        print(
            'Array {0}: {1}'.format(i, polydata_centers.GetPointData().GetArrayName(i))
        )

    for i in range(polydata_centers.GetCellData().GetNumberOfArrays()):
        print('Cell {0}: {1}'.format(i, polydata_centers.GetCellData().GetArrayName(i)))

    print('Array pts: {}'.format(polydata_centers.GetPoints().GetData().GetName()))

    glyph = vtk.vtkGlyph3D()
    if use_vertices:
        scale = 1
        my_polydata = vtk.vtkPolyData()
        my_vertices = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )

        my_vertices -= np.array([0.5, 0.5, 0])

        my_vertices = scale * my_vertices

        my_triangles = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

        set_polydata_vertices(my_polydata, my_vertices)
        set_polydata_triangles(my_polydata, my_triangles)

        glyph.SetSourceData(my_polydata)
    else:
        src = vtk.vtkPlaneSource()
        src.SetXResolution(size[0])
        src.SetYResolution(size[1])
        glyph.SetSourceConnection(src.GetOutputPort())

    glyph.SetInputData(polydata_centers)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyph.GetOutput())
    mapper.SetScalarModeToUsePointFieldData()

    mapper.SelectColorArray('colors')

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def square(scale=1):
    polydata = vtk.vtkPolyData()

    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )

    vertices -= np.array([0.5, 0.5, 0])

    vertices = scale * vertices

    triangles = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

    set_polydata_vertices(polydata, vertices)
    set_polydata_triangles(polydata, triangles)

    return get_actor_from_polydata(polydata)


def surface(vertices, faces=None, colors=None, smooth=None, subdivision=3):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    if colors is not None:
        polydata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(colors))

    if faces is None:
        tri = Delaunay(vertices[:, [0, 1]])
        faces = np.array(tri.simplices, dtype='i8')

    if faces.shape[1] == 3:
        triangles = np.empty((faces.shape[0], 4), dtype=np.int64)
        triangles[:, -3:] = faces
        triangles[:, 0] = 3
    else:
        triangles = faces

    if not triangles.flags['C_CONTIGUOUS'] or triangles.dtype != 'int64':
        triangles = np.ascontiguousarray(triangles, 'int64')

    cells = vtk.vtkCellArray()
    cells.SetCells(
        triangles.shape[0], numpy_support.numpy_to_vtkIdTypeArray(triangles, deep=True)
    )
    polydata.SetPolys(cells)

    clean_poly_data = vtk.vtkCleanPolyData()
    clean_poly_data.SetInputData(polydata)

    mapper = vtk.vtkPolyDataMapper()
    surface_actor = vtk.vtkActor()

    if smooth is None:
        mapper.SetInputData(polydata)
        surface_actor.SetMapper(mapper)

    elif smooth == 'loop':
        smooth_loop = vtk.vtkLoopSubdivisionFilter()
        smooth_loop.SetNumberOfSubdivisions(subdivision)
        smooth_loop.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_loop.GetOutputPort())
        surface_actor.SetMapper(mapper)

    elif smooth == 'butterfly':
        smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
        smooth_butterfly.SetNumberOfSubdivisions(subdivision)
        smooth_butterfly.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_butterfly.GetOutputPort())
        surface_actor.SetMapper(mapper)

    return surface_actor
