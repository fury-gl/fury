"""
================================
Collisions of particles in a box
================================

"""

import numpy as np
import sys
import math
from fury import window, actor, ui, utils, pick
import itertools
from vtk.util import numpy_support


def boundary_conditions():
    global vel, vcolors, xyz, num_particles
    num_vertices = vertices.shape[0]
    color_add = np.array([10, 0, 0], dtype='uint8')
    no_vertices_per_sphere = len(vertices)/num_particles
    sec = np.int(num_vertices / num_particles)
    for i,j in np.ndindex(num_particles, num_particles):
        if (i == j):
            continue
        distance = (((xyz[i, 0]-xyz[j, 0])**2) + ((xyz[i, 1]-xyz[j, 1])**2) + ((xyz[i, 1]-xyz[j, 1])**2))** 0.5
        if (distance <= (radii[i] + radii[j])):
            vel[i] = -vel[i]
            vel[j] = -vel[j]
            vcolors[i * sec: i * sec + sec] += color_add
         #   vcolors[j * sec: j * sec + sec] += color_add
            xyz[i] = xyz[i] + vel[i] * dt
            xyz[j] = xyz[j] + vel[j] * dt
            # vcolors[i * sec: i * sec + sec] -= color_add
            # vcolors[j * sec: j * sec + sec] -= color_add




    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + radii[:]) |
                          (xyz[:, 0] >= (0.5 * box_lx - radii[:]))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + radii[:]) | (xyz[:, 1] >= (0.5 * box_ly - radii[:]))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + radii[:]) | (xyz[:, 2] >= (0.5 * box_lz - radii[:]))),
                         - vel[:, 2], vel[:, 2])


global xyz, dt, steps, num_particles, vel, vcolors

num_particles = 200
box_lx = 50
box_ly = 50
box_lz = 50
steps = 1000
dt = 0.5

xyz = (box_lz * 0.75) * (np.random.rand(num_particles, 3) - 0.5)
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.zeros((num_particles, 3)) + np.array([0, 0.5, 0.3])
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
#scene.add(actor.axes(scale=(0.5*box_lx, 0.5*box_ly, 0.5*box_lz)))
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()
scene.add(box_actor)
tb = ui.TextBlock2D(bold=True)

# use itertools to avoid global variables
counter = itertools.count()

global vertices, vcolors
vertices = utils.vertices_from_actor(sphere_actor)
vcolors = utils.colors_from_actor(sphere_actor, 'colors')
no_vertices_per_sphere = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_sphere, axis=0)

def timer_callback(_obj, _event):
    global xyz, dt, steps, num_particles, vcolors, vel
    cnt = next(counter)
    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    boundary_conditions()
    xyz = xyz + vel * dt

    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    # Tell actor that memory is modified
    utils.update_actor(sphere_actor)
    showm.render()

    if cnt == steps:
        showm.exit()


scene.add(tb)
showm.add_timer_callback(True, 100, timer_callback)
showm.start()
