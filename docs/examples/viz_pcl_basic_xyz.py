from docs.experimental.tmp_pcl_fetcher import read_viz_dataset
from fury import actor, window
from fury.io import save_polydata
from fury.utils import (array_from_actor, get_actor_from_polydata,
                        set_polydata_colors, set_polydata_normals,
                        set_polydata_triangles, set_polydata_vertices)


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import vtk


scene = window.Scene()
scene.background((1, 1, 1))

scene.roll(-145)
scene.pitch(70)

pcd_fname = read_viz_dataset('sample.xyz')

pcd_data = np.loadtxt(pcd_fname, skiprows=1)

xyz = pcd_data[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

rgb = pcd_data[:, 3:] / 255
pcd.colors = o3d.utility.Vector3dVector(rgb)

pcd_actor = actor.dot(xyz, colors=rgb)

scene.add(pcd_actor)

scene.reset_camera()
scene.zoom(1.3)

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

# Normals estimation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=.1, max_nn=30))

normals = np.asarray(pcd.normals)

pcd_actor_colors = array_from_actor(pcd_actor, array_name='colors')
pcd_actor_colors[:, :] = (normals + 1) / 2 * 255

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

# NOTE: This might take some time
pcd.orient_normals_consistent_tangent_plane(10)

normals = np.asarray(pcd.normals)

pcd_actor_colors[:, :] = (normals + 1) / 2 * 255

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

scene.clear()

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, den = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

mesh_vertices = rec_mesh.vertices
mesh_triangles = rec_mesh.triangles
mesh_normals = rec_mesh.vertex_normals

polydata = vtk.vtkPolyData()
set_polydata_vertices(polydata, mesh_vertices)
set_polydata_triangles(polydata, mesh_triangles)
set_polydata_normals(polydata, mesh_normals)
set_polydata_colors(polydata, np.asarray(rec_mesh.vertex_colors) * 255)
mesh_actor = get_actor_from_polydata(polydata)

scene.add(mesh_actor)

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

den = np.asarray(den)
den_colors = plt.get_cmap('viridis')(
    (den - den.min()) / (den.max() - den.min()))
den_colors = den_colors[:, :3]

set_polydata_colors(polydata, den_colors * 255)

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

scene.clear()

verts_to_remove = den < np.quantile(den, .02)
rec_mesh.remove_vertices_by_mask(verts_to_remove)

polydata = vtk.vtkPolyData()
set_polydata_vertices(polydata, rec_mesh.vertices)
set_polydata_triangles(polydata, rec_mesh.triangles)
set_polydata_normals(polydata, rec_mesh.vertex_normals)
set_polydata_colors(polydata, np.asarray(rec_mesh.vertex_colors) * 255)
mesh_actor = get_actor_from_polydata(polydata)

scene.add(mesh_actor)

# TODO: Replace with SS
#window.show(scene, reset_camera=False)

#save_polydata(polydata, 'glyptotek.vtk')
