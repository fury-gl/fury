"""
======================================================================
Motion of a charged particle in a combined magnetic and electric field
======================================================================

A charged particle follows a curved path in a magnetic field.
In an electric field, the particle tends to accelerate in a direction parallel/
antiparallel to the electric field depending on the nature of charge on the
particle.
In a combined electric and magnetic field, the particle moves along a helical
path.

In this animation, there's a magnetic and an electric field present in +x
direction under whose influence the charged particle follows a helical path.

"""

###############################################################################
# Importing necessary modules
from fury import window, actor, utils, ui
import numpy as np
import itertools


###############################################################################
# Variable(s) and their description-
# radius_particle: radius of the point that will represent the particle
#                  (default = 0.08)
# radius_path: radius of the points that will render the path of the particle
#                  (default = 0.04)
# u: initial velocity of the particle
#    (default = 0.09)
# a: acceleration of the particle along +x (due to the electric field)
# t: time
#    (default = 0)
# dt: value by whic
# k: wavenumber = 2*pi/wavelength
# t: time (default time i.e. time at beginning  = 0)
# dt: value by which t is incremented for each call of timer_callback
# w: angular frequency (default = 0.1)
# d: phase angle (default = 0.002)

radius_particle = 0.08
radius_path = 0.04
u = 0.09
a = 0.004
t = 0
dt = 0.09
w = 0.1
d = 0.002


###############################################################################
# Creating a scene object and configuring the camera's position
scene = window.Scene()
scene.zoom(1.2)
scene.set_camera(position=(10, 12.5, 19), focal_point=(3.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 0.0))
showm = window.ShowManager(scene,
                           size=(800, 600), reset_camera=True,
                           order_transparent=True)
showm.initialize()


###############################################################################
# Creating a blue colored arrow to show the direction of propagation of
# electromagnetic wave
color_arrow = window.colors.blue  # color of the arrow can be manipulated
centers = np.array([[0, 0, 0]])
directions = np.array([[1, 0, 0]])
heights = np.array([8])
arrow_actor = actor.arrow(centers, directions, color_arrow, heights,
                          resolution=20, tip_length=0.06, tip_radius=0.012,
                          shaft_radius=0.005)
scene.add(arrow_actor)

###############################################################################
# Initializing the initial coordinates of the particle
x = u*t + 0.5*a*(t**2)
y = np.sin(w*t + d)
z = np.cos(w*t + d)

###############################################################################
# Initializing point actor which will represent the charged particle

color_particle = window.colors.red  # color of particle can be manipulated
pts = np.array([[x, y, z]])
line_actor = actor.point(pts, color_particle, point_radius=radius_particle)
scene.add(line_actor)

vertices = utils.vertices_from_actor(line_actor)
vcolors = utils.colors_from_actor(line_actor, 'colors')
no_vertices_per_point = len(vertices)
initial_vertices = vertices.copy() - \
    np.repeat(pts, no_vertices_per_point, axis=0)


###############################################################################
# Initializing text box to display the name of the animation
tb = ui.TextBlock2D(bold=True, position=(100, 90))
m1 = "Motion of a charged particle in a "
m2 = "combined electric and magnetic field"
tb.message = m1 + m2
scene.add(tb)

###############################################################################
# Initializing counter
counter = itertools.count()

###############################################################################
# end is used to decide when to end the animation
end = 500

###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The wave is rendered here.


def timer_callback(_obj, _event):
    global pts, t, dt
    t += dt
    x = u*t + 0.5*a*(t**2)
    y = np.sin(10*w*t + d)
    z = np.cos(10*w*t + d)
    pts = np.array([[x, y, z]])

    vertices[:] = initial_vertices + \
        np.repeat(pts, no_vertices_per_point, axis=0)

    # Plotting the path followed by the particle
    pts_path = pts
    color_path = window.colors.cyan
    path_actor = actor.point(pts_path, color_path, point_radius=radius_path)
    scene.add(path_actor)

    utils.update_actor(line_actor)

    cnt = next(counter)
    showm.render()

    # to end the animation
    if cnt == end:
        showm.exit()

###############################################################################
# Run every 15 milliseconds


showm.add_timer_callback(True, 15, timer_callback)

interactive = False
if interactive:
    showm.start()
window.record(showm.scene, size=(800, 600), out_path="helical_motion.png")
