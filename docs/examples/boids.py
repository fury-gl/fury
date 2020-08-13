"""
================================
Flock simulation in a box
================================

This is an example of boids in a box using FURY.
"""

##############################################################################
# Explanation:

import numpy as np
import math
from fury import window, actor, ui, utils, primitive
import itertools


global xyz, directions
num_particles = 3
steps = 1000
dt = 0.05
# xyz = np.random.rand(num_particles, 3) *2
vel = np.random.rand(num_particles, 3)
colors = np.array([[0.5, 0.5, 0.5],
                    [0, 1, 0.],
                    [0.5, 0.5, 0.5]])
xyz = 0 *np.array([[10, 0, 0.],
                    [10 + 3, 0 , 0.],
                    [13 + 3, 0, 0.]])
directions = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0.]])

scene = window.Scene()
arrow_actor = actor.arrow(centers=xyz,
                          directions=directions, colors=colors, heights=3,
                          resolution=10, vertices=None, faces=None)
scene.add(arrow_actor)
axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()

tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(arrow_actor)
vcolors = utils.colors_from_actor(arrow_actor, 'colors')
no_vertices_per_sphere = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_sphere, axis=0)

scene.zoom(0.8)

def timer_callback(_obj, _event):
    global xyz, directions
    turnfraction = 0.01
    cnt = next(counter)
    dst = 2
    angle = 2 * np.pi * turnfraction * cnt
    x = dst * np.cos(angle)
    y = dst * np.sin(angle)
    directions = np.array([[x, y, 0],
                       [x, y, 0],
                       [x, y, 0.]])

    xyz = np.array([[x, y, 0.],
                       [x, y, 0.],
                       [x, y, 0.]])
    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    utils.update_actor(arrow_actor)
    scene.reset_clipping_range()
    showm.render()
    if cnt == steps-1:
        showm.exit()

scene.add(tb)
showm.add_timer_callback(True, 50, timer_callback)
showm.start()
