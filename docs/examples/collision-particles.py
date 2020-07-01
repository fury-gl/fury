"""
================================
Collisions of particles in a box
================================

"""

import numpy as np
import sys
import math
from fury import window, actor, ui
import itertools
from vtk.util import numpy_support

global xyz, box_lx, box_ly, box_lz, dt, steps

num_particles = 200
box_lx = 50
box_ly = 50
box_lz = 50
steps = 1000
dt = 0.5

xyz = (box_lz * 0.75) * (np.random.rand(num_particles, 3) - 0.5)
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.random.rand(num_particles, 3)
radii = np.random.rand(num_particles) + 0.01

scene = window.Scene()

box_centers = np.array([[0, 0, 0]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[1, 1, 1, 0.2]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                      size=(box_lx, box_ly, box_lz),
                      heights=1, vertices=None, faces=None)


box_actor.GetProperty().SetLineWidth(1)
box_actor.GetProperty().SetOpacity(1)

sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)
scene.add(sphere_actor)
scene.add(actor.axes(scale=(0.5*box_lx, 0.5*box_ly, 0.5*box_lz)))
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
no_vertices_per_sphere = len(all_vertices)/num_particles
initial_vertices = all_vertices.copy() - np.repeat(xyz, no_vertices_per_sphere, axis=0)

def timer_callback(_obj, _event):
    global vel, xyz, box_lx, box_ly, box_lz, dt, steps
    cnt = next(counter)
    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    for i,j in np.ndindex(num_particles, num_particles):
        distance = (((xyz[i, 0]-xyz[j, 0])**2) + ((xyz[i, 1]-xyz[j, 1])**2) + ((xyz[i, 1]-xyz[j, 1])**2))** 0.5
        if (i == j):
            continue
        if (distance <= (radii[i] + radii[j])):
            vel[i, :] = - vel[i, :]
            vel[j, :] = - vel[j, :]

    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + radii[:]) | (xyz[:, 0] >= (0.5 * box_lx - radii[:]))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + radii[:]) | (xyz[:, 1] >= (0.5 * box_ly - radii[:]))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + radii[:]) | (xyz[:, 2] >= (0.5 * box_lz - radii[:]))),
                         - vel[:, 2], vel[:, 2])

    xyz = xyz + vel * dt
    all_vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    set_vertices(sphere_actor, all_vertices)
    modified(sphere_actor)
    showm.render()

    if cnt == steps:
        showm.exit()


scene.add(tb)
showm.add_timer_callback(True, 10, timer_callback)
showm.start()
