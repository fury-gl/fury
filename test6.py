from fury import actor, window
import numpy as np
import vtk



def dots(points, color=(1, 0, 0), opacity=1, dot_size=5):
    """Create one or more 3d points.

    Parameters
    ----------
    points : ndarray, (N, 3)
    color : tuple (3,)
    opacity : float, optional
        Takes values from 0 (fully transparent) to 1 (opaque)
    dot_size : int

    Returns
    --------
    vtkActor

    See Also
    ---------
    fury.actor.point

    """
    if points.ndim == 2:
        points_no = points.shape[0]
    else:
        points_no = 1

    polyVertexPoints = vtk.vtkPoints()
    polyVertexPoints.SetNumberOfPoints(points_no)
    aPolyVertex = vtk.vtkPolyVertex()
    aPolyVertex.GetPointIds().SetNumberOfIds(points_no)

    cnt = 0
    if points.ndim > 1:
        for point in points:
            polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
            aPolyVertex.GetPointIds().SetId(cnt, cnt)
            cnt += 1
    else:
        polyVertexPoints.InsertPoint(cnt, points[0], points[1], points[2])
        aPolyVertex.GetPointIds().SetId(cnt, cnt)
        cnt += 1

    aPolyVertexGrid = vtk.vtkUnstructuredGrid()
    aPolyVertexGrid.Allocate(1, 1)
    aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(),
                                   aPolyVertex.GetPointIds())

    aPolyVertexGrid.SetPoints(polyVertexPoints)
    aPolyVertexMapper = vtk.vtkDataSetMapper()
    aPolyVertexMapper.SetInputData(aPolyVertexGrid)
    aPolyVertexActor = vtk.vtkActor()
    aPolyVertexActor.SetMapper(aPolyVertexMapper)

    aPolyVertexActor.GetProperty().SetColor(color)
    aPolyVertexActor.GetProperty().SetOpacity(opacity)
    aPolyVertexActor.GetProperty().SetPointSize(dot_size)
    return aPolyVertexActor


# start_pos = np.array([0, 0, 0])
start_pos = np.array([[0, 0, 0], [1, 1, 1]])
dot_size = 1
color = [1, 0, 0]
# color = [[1, 0, 0], [1, 0, 0]]
# color = np.array([[1, 0, 0], [1, 0, 0]])
# color = ([1, 0, 0], [1, 0, 0])

# color = np.random.rand(2, 3)

scene = window.Scene()
# c = actor.dots(start_pos, color, dot_size)
c = dots(start_pos, color, dot_size)
scene.add(c)
window.show(scene)