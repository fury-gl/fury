from docs.experimental.tmp_pcl_fetcher import read_viz_dataset
from fury import window


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import vtk


pcl_fname = read_viz_dataset('TLS_kitchen.ply')
pcd = o3d.io.read_point_cloud(pcl_fname)

xyz = np.asarray(pcd.points)
rgb = np.ones((len(xyz), 3)) * 40

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=.1, max_nn=30))

"""
plane_model, inliers = pcd.segment_plane(distance_threshold=.01, ransac_n=3,
                                         num_iterations=1000)
#inlier_cloud = pcd.select_by_index(inliers)
#outlier_cloud = pcd.select_by_index(inliers, invert=True)
#inlier_cloud.paint_uniform_color([1, 0, 0])
#outlier_cloud.paint_uniform_color([.5, .5, .5])
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
rgb = np.ones((len(xyz), 3)) * 40
rgb[inliers, :] = [255, 0, 0]
"""

"""
# Clustering with DBSCAN
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=.05, min_points=10,
                                         print_progress=True))
#labels = np.array(pcd.cluster_dbscan(eps=.05, min_points=10))

max_label = labels.max()
cols = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
cols[labels < 0] = 0
rgb = cols[:, :3] * 255
pcd.colors = o3d.utility.Vector3dVector(cols[:, :3])
"""

segment_models = {}
segments = {}

max_plane_idx = 20

cmap = plt.get_cmap('tab20')
ransac_thr = .01

rest = pcd
xyz = []
rgb = []
for i in range(max_plane_idx):
    color = list(cmap(i)[:3])
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=ransac_thr, ransac_n=3, num_iterations=1000)
    inlier_cloud = rest.select_by_index(inliers)
    pnts = np.asarray(inlier_cloud.points)
    ransac_num_pnts = len(pnts)
    labels = np.array(inlier_cloud.cluster_dbscan(eps=ransac_thr * 10,
                                                  min_points=10))
    clusters_ids = np.unique(labels)
    ppc = [len(np.where(labels == j)[0]) for j in clusters_ids]
    biggest_cluster = clusters_ids[np.argwhere(ppc == np.max(ppc))[0][0]]
    #segments[i] = inlier_cloud
    #segments[i].paint_uniform_color(color)
    rest = rest.select_by_index(inliers, invert=True)
    rest += inlier_cloud.select_by_index(list(np.where(
        labels != biggest_cluster)[0]))
    inlier_cloud = inlier_cloud.select_by_index(list(np.where(
        labels == biggest_cluster)[0]))
    pnts = np.asarray(inlier_cloud.points)
    cols = np.repeat(np.array([color]), len(pnts), axis=0) * 255
    xyz.extend(pnts)
    rgb.extend(cols)
    print('{:02d}/{} done. {: 7d} points found by RANSAC. '
          '{: 7d} points retained by DBSCAN.'.format(i + 1, max_plane_idx,
                                                 ransac_num_pnts, len(pnts)))
labels = np.array(rest.cluster_dbscan(eps=.05, min_points=5))
max_label = labels.max()
pnts = np.asarray(rest.points)
#cols = np.ones((len(pnts), 3)) * 40
cols = cmap(labels / max_label if max_label > 0 else 1)
cols[labels < 0] = 0
cols = cols[:, :3] * 255
xyz.extend(pnts)
rgb.extend(cols)
xyz = np.asarray(xyz)
rgb = np.asarray(rgb)

"""
o3d.visualization.draw_geometries([segments[i] for i in range(
    max_plane_idx)] + [rest])
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

scene.add(pcl_actor)

scene.roll(-160)
scene.pitch(85)

scene.reset_camera()
scene.reset_clipping_range()

#scene.zoom(2)
#scene.zoom(2.1)
scene.zoom(2.4)

window.show(scene, reset_camera=False)
#o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)
