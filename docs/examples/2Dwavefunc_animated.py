"""
===============================================
Animating a time-varying 2D wave function
===============================================

This is a simple demonstration of how one can visualize
time-varying 2D wave functions using FURY.

Can try plotting these in update_coordinates function-
#Z = (X**2 - Y**2)/(X**2 + Y**2)**0.5*np.cos(np.pi*X-rate*incre)
#Z = np.sin(X**2)*np.cos(Y**2-rate*incre)#*np.sin(rate*incre)
#Z = np.sin(X**2-Y**2)*np.sin(time)

"""

###############################################################################
# Importing necessary modules

from fury import window, actor, ui, utils, colormap
import numpy as np
import itertools

###############################################################################
# The following function is used to scale the Z values(height) between 0-1
# which will then be mapped in RGB colors according to colormap


def scale(arr):
    arr = np.array(arr)
    maxima = np.max(arr)
    minima = np.min(arr)
    return (arr - maxima)/(maxima - minima)

###############################################################################
# The following function updates and returns the coordinates of the points
# which are being used to plot the 2d function. Kindly note that only the z
# coordinate is being modified with time as only the z coordinate is a
# function of time


def update_coordinates(lower_xbound=-1, upper_xbound=1, lower_ybound=-1,
                       upper_ybound=1, npoints=100, time=0, incre=0):
    x = np.linspace(lower_xbound, upper_xbound, npoints)
    y = np.linspace(lower_ybound, upper_ybound, npoints)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    # manipulate rate to change the speed of the waves; (default = 0.025)
    rate = 0.025
    # Z is the function F i.e. F(x, y, t)
    Z = 0.34*np.sin(2*np.pi*(X))*np.cos(2*np.pi*(Y-rate*incre))*np.cos(time)
    v = scale(Z)
    colors = colormap.create_colormap(v, name='PuBu')
    xyz = np.array([(a, b, c) for (a, b, c) in zip(X, Y, Z)])
    return xyz, colors

###############################################################################
# Creating a scene object and configuring the camera's position


scene = window.Scene()
scene.zoom(10.5)
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
# upper_ybound: Upper bound of the y values in which the function is plotted
#                (default = 1)
# npoints: For high quality rendering, keep the number high but kindly note
#          that higher values for npoints slows down the animation
#          (default = 200)

time = 0
dt = 0.1
lower_xbound = -1
upper_xbound = 1
lower_ybound = -1
upper_ybound = 1
npoints = 200


###############################################################################
# xyz are the coordinates of the points that'll be used to plot the function
# colors refers to the colormap that'll be used
xyz, colors = update_coordinates(lower_xbound, upper_xbound, lower_ybound,
                                 upper_ybound, npoints, time=time)

###############################################################################
# Initializing point and axes actors and adding them to the scene
point_actor = actor.surface(xyz, colors=colors)
axes_actor = actor.axes(scale=(1.5, 1.5, 1.5))
scene.add(axes_actor)
scene.add(point_actor)

###############################################################################
# Initializing text box to print the 2D function which is being rendered
tb = ui.TextBlock2D(bold=True, position=(150, 60))
tb.message = "z = F(x, y, t) = 0.34*sin(2*pi*x)*cos(2*p*y)*cos(t)"
scene.add(tb)

###############################################################################
# Initializing showm and counter
showm.initialize()
counter = itertools.count()

###############################################################################
# will be used for storing the point_actor's attributes
vertices = utils.vertices_from_actor(point_actor)
vcolors = utils.colors_from_actor(point_actor, 'colors')
no_vertices_per_point = len(vertices)/npoints**2
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_point, axis=0)

###############################################################################
# end is used to decide when to end the animation
end = 2000

###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The 2D function is rendered here


def timer_callback(_obj, _event):
    global xyz, time, colors
    time += dt
    cnt = next(counter)
    xyz, colors[:] = update_coordinates(lower_xbound, upper_xbound,
                                        lower_ybound, upper_ybound,
                                        npoints, time=time, incre=cnt)
    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_point, axis=0)
    utils.update_actor(point_actor)
    scene.reset_clipping_range()
    showm.render()

# to end the animation
    if cnt == end:
        showm.exit()

###############################################################################
# Run every 50 milliseconds


showm.add_timer_callback(True, 50, timer_callback)

interactive = True
if interactive:
    showm.start()

window.record(showm.scene, size=(600, 600), out_path="2Dwave_func.png")
