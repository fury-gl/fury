"""
===============
===============

"""


import numpy as np
import sys
import math
from fury import window, actor, ui
import itertools
from numba import numpy_support
from vtk.util import numpy_support
import itertools

num_particles = 10
box_lx = box_ly = 10
box_lz =10

global force, velocity, xyz
force = np.random.rand(num_particles, 3)
# force = np.ones((10, 1))
velocity = np.random.rand(num_particles, 3)
xyz = 10 * np.random.rand(num_particles, 3)
colors = np.random.rand(num_particles, 3)
radii = np.random.rand(num_particles)

scene = window.Scene()
box_centers = np.array([[0, 0, 0.]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[1, 0, 0, 0.2]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                    size=(box_lx, box_ly, box_lz),
                    heights=2, vertices=None, faces=None)
box_actor.GetProperty().SetRepresentationToWireframe()
box_actor.GetProperty().SetLineWidth(10)

sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)
scene.add(sphere_actor)
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()
scene.add(box_actor)
tb = ui.TextBlock2D(bold=True)

# use itertools to avoid global variables

counter = itertools.count()


def modified(act):
    act.GetMapper().GetInput().GetPoints().GetData().Modified()
    act.GetMapper().GetInput().ComputeBounds()


def set_vertices(act, num_arr):
    vtk_num_array = numpy_support.numpy_to_vtk(num_arr)
    act.GetMapper().GetInput().GetPoints().SetData(vtk_num_array)


def get_vertices(act):
    all_vertices = np.array(numpy_support.vtk_to_numpy(
        act.GetMapper().GetInput().GetPoints().GetData()))
    return all_vertices


global all_vertices
all_vertices = get_vertices(sphere_actor)
initial_vertices = all_vertices.copy()
no_vertices_per_sphere = len(all_vertices)/num_particles
dt = 0.5

steps = 100


def timer_callback(_obj, _event):
    global force, velocity, xyz
    cnt = next(counter)
    tb.message = "Let's count up to 100 and exit :" + str(cnt)
    velocity = velocity + force * (0.5 * dt)
    xyz = xyz + velocity * dt
    all_vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    set_vertices(sphere_actor, all_vertices)
    modified(sphere_actor)
    showm.render()
    if cnt == steps:
        showm.exit()


scene.add(tb)
showm.add_timer_callback(True, 200, timer_callback)
showm.start()
