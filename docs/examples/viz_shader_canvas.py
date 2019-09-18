from fury import window
from fury.utils import get_actor_from_polydata, set_polydata_triangles, set_polydata_vertices
from scipy.spatial import Delaunay
from vtk.util import numpy_support


import numpy as np
import vtk


def cube():
    my_polydata = vtk.vtkPolyData()

    my_vertices = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0]])

    my_vertices -= 0.5

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

    set_polydata_vertices(my_polydata, my_vertices)
    set_polydata_triangles(my_polydata, my_triangles)
    return get_actor_from_polydata(my_polydata)


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


def square(scale=1):
    my_polydata = vtk.vtkPolyData()

    my_vertices = np.array([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0]])

    my_vertices -= np.array([0.5, 0.5, 0])

    my_vertices = scale * my_vertices

    my_triangles = np.array([[0, 1, 2],
                             [2, 3, 0]], dtype='i8')

    set_polydata_vertices(my_polydata, my_vertices)
    set_polydata_triangles(my_polydata, my_triangles)

    return get_actor_from_polydata(my_polydata)


def surface(vertices, faces=None, colors=None, smooth=None, subdivision=3):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices))
    triangle_poly_data = vtk.vtkPolyData()
    triangle_poly_data.SetPoints(points)

    if colors is not None:
        triangle_poly_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(colors))

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
    cells.SetCells(triangles.shape[0], numpy_support.numpy_to_vtkIdTypeArray(triangles, deep=True))
    triangle_poly_data.SetPolys(cells)

    clean_poly_data = vtk.vtkCleanPolyData()
    clean_poly_data.SetInputData(triangle_poly_data)

    mapper = vtk.vtkPolyDataMapper()
    surface_actor = vtk.vtkActor()

    if smooth is None:
        mapper.SetInputData(triangle_poly_data)
        surface_actor.SetMapper(mapper)

    elif smooth == "loop":
        smooth_loop = vtk.vtkLoopSubdivisionFilter()
        smooth_loop.SetNumberOfSubdivisions(subdivision)
        smooth_loop.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_loop.GetOutputPort())
        surface_actor.SetMapper(mapper)

    elif smooth == "butterfly":
        smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
        smooth_butterfly.SetNumberOfSubdivisions(subdivision)
        smooth_butterfly.SetInputConnection(clean_poly_data.GetOutputPort())
        mapper.SetInputConnection(smooth_butterfly.GetOutputPort())
        surface_actor.SetMapper(mapper)

    return surface_actor
