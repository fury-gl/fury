from fury import actor, window
from docs.experimental.tmp_pcl_fetcher import read_viz_dataset


import numpy as np
import vtk


pcl_fname = read_viz_dataset('sample.xyz')

point_cloud = np.loadtxt(pcl_fname, skiprows=1)

xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:]

## VTK
# Create the geometry of a point (the coordinate)
points = vtk.vtkPoints()
# Create the topology of the point (a vertex)
vertices = vtk.vtkCellArray()
# Setup colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName('colors')
# Add points
for i in range(0, len(xyz)):
    p = xyz[i]
    id = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(id)
    colors.InsertNextTuple3(rgb[i][0], rgb[i][1], rgb[i][2])
# Create a polydata object
point = vtk.vtkPolyData()
# Set the points and vertices we created as the geometry and topology of the
# polydata
point.SetPoints(points)
point.SetVerts(vertices)
point.GetPointData().SetScalars(colors)
point.Modified()
# Visualize
mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(point)
else:
    mapper.SetInputData(point)
## ACTOR
# Create an actor
pcl_actor = vtk.vtkActor()
pcl_actor.SetMapper(mapper)
#pcl_actor.GetProperty().SetPointSize(10)

scene = window.Scene()

#scene.add(actor.dots(xyz))
scene.add(pcl_actor)

window.show(scene)
