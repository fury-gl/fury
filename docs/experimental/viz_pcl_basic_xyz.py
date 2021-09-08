from docs.experimental.tmp_pcl_fetcher import read_viz_dataset
from fury import window
from vtk.util import numpy_support


import numpy as np
import open3d as o3d
import vtk


pcl_fname = read_viz_dataset('sample.xyz')

point_cloud = np.loadtxt(pcl_fname, skiprows=1)

xyz = point_cloud[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

rgb = point_cloud[:, 3:]
pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

# Normals estimation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=.1, max_nn=30))

ijk = np.asarray(pcd.normals)
rgb = (ijk + 1) / 2 * 255
#pcd.colors = o3d.utility.Vector3dVector((ijk + 1) / 2)

## VTK
# Create the geometry of a point (the coordinate)
points = vtk.vtkPoints()
# Create the topology of the point (a vertex)
vertices = vtk.vtkCellArray()
# Setup colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName('colors')
# Setup normals
#normals = vtk.vtkUnsignedCharArray()
#normals.SetNumberOfComponents(3)
# Add points
for i in range(0, len(xyz)):
    p = xyz[i]
    id = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(id)
    colors.InsertNextTuple3(rgb[i][0], rgb[i][1], rgb[i][2])
    #normals.InsertNextTuple3(ijk[i][0], ijk[i][1], ijk[i][2])
# Create a polydata object
point = vtk.vtkPolyData()
# Set the points and vertices we created as the geometry and topology of the
# polydata
point.SetPoints(points)
point.SetVerts(vertices)
point.GetPointData().SetScalars(colors)
#normals = numpy_support.numpy_to_vtk(ijk, deep=True)
#point.GetPointData().SetNormals(normals)
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
scene.background((1, 1, 1))

scene.add(pcl_actor)

scene.roll(-145)
scene.pitch(70)

scene.reset_camera()
scene.reset_clipping_range()

#scene.zoom(2)
#scene.zoom(2.1)
#scene.zoom(2.4)

window.show(scene, reset_camera=False)
#o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)
