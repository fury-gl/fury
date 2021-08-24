from fury import actor, window
from docs.experimental.tmp_pcl_fetcher import read_viz_dataset


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import vtk


#pcl_fname = read_viz_dataset('sample.xyz')
pcl_fname = read_viz_dataset('NZ19_Wellington_voxel-center_2.xyz')

#point_cloud = np.loadtxt(pcl_fname, skiprows=1)
point_cloud = np.loadtxt(pcl_fname, delimiter=';', skiprows=1)

xyz = point_cloud[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

#rgb = point_cloud[:, 3:]
rgb = np.ones((len(xyz), 3)) * 255
#pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

# Normals estimation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=.1, max_nn=30))

normals = np.asarray(pcd.normals)
rgb = (normals + 1) / 2 * 255
#pcd.colors = o3d.utility.Vector3dVector((normals + 1) / 2)

"""
plane_model, inliers = pcd.segment_plane(distance_threshold=.1, ransac_n=3,
                                         num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([.5, .5, .5])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
"""

"""
# Clustering with DBSCAN
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=.027, min_points=10,
                                         print_progress=True))

max_label = labels.max()
cols = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
cols[labels < 0] = 0

rgb = cols[:, :3] * 255
#pcd.colors = o3d.utility.Vector3dVector(cols[:, :3])
"""

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
scene.background((1, 1, 1))

#scene.add(actor.dots(xyz))
scene.add(pcl_actor)

scene.roll(-145)
scene.pitch(70)
scene.reset_camera()
scene.reset_clipping_range()
#scene.zoom(2)

window.show(scene, reset_camera=False)
#o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)
