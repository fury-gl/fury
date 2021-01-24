"""
===============================================
Animating a time-varying 2D wave function
===============================================

This is a simple demonstration of how one can visualize
time-varying 2D wave functions using FURY.
An example of a 2D time varying wave function-
z = F(x, y, t) = 0.34*sin(10x)*cos(10x)*cos(t)

Other examples-
#Z = (X**2 - Y**2)/(X**2 + Y**2)**0.5*np.cos(t)
#Z = (X**2 + Y**2 )*np.sin(t)
#Z = np.sin(X**2)*np.cos(Y**2)*np.sin(t) #flapping wings
#Z = np.sin(X**2-Y**2)*np.sin(t)

"""

###############################################################################
# Importing necessary modules

from fury import window, actor, ui, utils
import numpy as np
import itertools

###############################################################################
# The following function updates and returns the coordinates of the points
# which are being used to plot the 2d function. Kindly note that only the z
# coordinate is being modified time as only the z coordinate is a function of
# time.

def coordinates(lower_xbound=-1, upper_xbound=1, lower_ybound=-1,
                upper_ybound=1, npoints=100, t=0):
    x = np.linspace(lower_xbound, upper_xbound, npoints)
    y = np.linspace(lower_ybound, upper_ybound, npoints)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = 0.34*np.sin(X*10)*np.cos(10*Y)*np.cos(t)
    xyz = np.array([(a, b, c) for (a, b, c) in zip(X, Y, Z)])
    return xyz

###############################################################################
# Creating a scene object and configuring the camera's position

scene = window.Scene()
scene.zoom(9.5)
steps = 500
scene.set_camera(position=(60, 40, 30), focal_point=(0.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 1.0))
showm = window.ShowManager(scene,
                           size=(600, 600), reset_camera=True,
                           order_transparent=True)

###############################################################################
# Variables and their usage-
# time: initial value of the time variable i.e. value of the time variable at
#        the beginning of the program; (default = 0)
# dt: amount by which the "time" variable is incremented for every iteration
#      of timer_callback function; (default = 0.1)
# lower_xbound: lower bound of the x values in which the function is plotted
#               (default = -1)
# upper_xbound: Upper bound of the x values in which the function is plotted
#                (default = 1)
# lower_ybound: lower bound of the y values in which the function is plotted
#               (default = -1)
# upper_xbound: Upper bound of the y values in which the function is plotted
#                (default = 1)
# npoints: It's basically the number of equidistant values between lower
#          and upper bounds that'll be used to plot the function.
#          Alternatively, it can also be thought of as the square root of the
#          total number of points that'll be used to plot the function.
#          For higher quality graphing, keep the number higher but do note
#          that higher values for n points significantly slows the output
#          (default = 100)

time = 0
dt = 0.12
lower_xbound = -1
upper_xbound = 1
lower_ybound = -1
upper_ybound = 1
npoints = 100

xyz = coordinates(lower_xbound, upper_xbound, lower_ybound, upper_ybound,
                 npoints, t=time)



colors = window.colors.orange
radius = 0.05

point_actor = actor.point(xyz, colors, point_radius=radius)
axes_actor = actor.axes(scale=(2, 2, 2))
scene.add(axes_actor)
scene.add(point_actor)

tb = ui.TextBlock2D(bold=True, position=(150, 90))
showm.initialize()
counter = itertools.count()


vertices = utils.vertices_from_actor(point_actor)
vcolors = utils.colors_from_actor(point_actor, 'colors')
no_vertices_per_point = len(vertices)/npoints**2
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_point, axis=0)

tb.message = "f(x, y, t) = 0.24*sin(x)*cos(y)*cos(t)"


def timer_callback(_obj, _event):
    global xyz
    global time
    time += dt
    cnt = next(counter)
    xyz = coordinates(lower_xbound, upper_xbound, lower_ybound, upper_ybound,
                     npoints, t=time)
    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_point, axis=0)
    utils.update_actor(point_actor)
    scene.reset_clipping_range()
    showm.render()

    if cnt == steps:
        showm.exit()

scene.add(tb)
showm.add_timer_callback(True, 5, timer_callback)

interactive = True
if interactive:
    showm.start()