from docs.experimental.tmp_pcl_fetcher import read_viz_dataset
from fury import window
from fury.utils import (array_from_actor, get_actor_from_polydata,
                        set_polydata_colors, set_polydata_normals,
                        set_polydata_triangles, set_polydata_vertices)


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import vtk


# TODO: Move to actor
def point_cloud(points, colors=None, normals=None, affine=None, point_size=1):
    # TODO: if affine is None
    # Create the geometry of a point (the coordinate)
    vtk_vertices = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vtk_faces = vtk.vtkCellArray()
    # Add points
    for i in range(len(points)):
        p = points[i]
        id = vtk_vertices.InsertNextPoint(p)
        vtk_faces.InsertNextCell(1)
        vtk_faces.InsertCellPoint(id)
    # Create a polydata object
    polydata = vtk.vtkPolyData()
    # Set the vertices and faces we created as the geometry and topology of the
    # polydata
    polydata.SetPoints(vtk_vertices)
    polydata.SetVerts(vtk_faces)
    if colors is not None:
        set_polydata_colors(polydata, colors)
    if normals is not None:
        set_polydata_normals(polydata, normals)
    polydata.Modified()
    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)
    # Create an actor
    pcl_actor = vtk.vtkActor()
    pcl_actor.SetMapper(mapper)
    pcl_actor.GetProperty().SetPointSize(point_size)
    return pcl_actor


scene = window.Scene()
scene.background((1, 1, 1))

scene.roll(-145)
scene.pitch(70)

pcd_fname = read_viz_dataset('sample.xyz')

pcd_data = np.loadtxt(pcd_fname, skiprows=1)

xyz = pcd_data[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

rgb = pcd_data[:, 3:]
pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

pcd_actor = point_cloud(xyz, colors=rgb)

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

verts_to_remove = den < np.quantile(den, .01)
rec_mesh.remove_vertices_by_mask(verts_to_remove)

polydata = vtk.vtkPolyData()
set_polydata_vertices(polydata, rec_mesh.vertices)
set_polydata_triangles(polydata, rec_mesh.triangles)
set_polydata_normals(polydata, rec_mesh.vertex_normals)
set_polydata_colors(polydata, np.asarray(rec_mesh.vertex_colors) * 255)
mesh_actor = get_actor_from_polydata(polydata)

scene.add(mesh_actor)

# TODO: Replace with SS
window.show(scene, reset_camera=False)
