"""
===============================================
Animating a time-varying 2D wave function
===============================================

This is a simple demonstration of how one can visualize
time-varying 2D wave functions using FURY.
2D time varying wave function used in this animation-
z = F(x, y, t) = 0.24*sin(10x)*cos(10y)*cos(t)

Try these in update_coordinates function-
#Z = (X**2 - Y**2)/(X**2 + Y**2)**0.5*np.cos(t)
#Z = (X**2 + Y**2 )*np.sin(t)
#Z = np.sin(X**2)*np.cos(Y**2)*np.sin(t) look like flapping wings
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
# coordinate is being modified with time as only the z coordinate is a
# function of time.

def update_coordinates(lower_xbound=-1, upper_xbound=1, lower_ybound=-1,
                upper_ybound=1, npoints=100, t=0):
    x = np.linspace(lower_xbound, upper_xbound, npoints)
    y = np.linspace(lower_ybound, upper_ybound, npoints)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    # Z is the function F i.e. F(x, y, t)
    Z = 0.34*np.sin(X*10)*np.cos(10*Y)*np.cos(t)
    xyz = np.array([(a, b, c) for (a, b, c) in zip(X, Y, Z)])
    return xyz

###############################################################################
# Creating a scene object and configuring the camera's position.

scene = window.Scene()
scene.zoom(9.5)
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
# npoints: It's basically the number of equidistant values between lower
#          and upper bounds that'll be used to plot the function.
#          Alternatively, it can also be thought of as the square root of the
#          total number of points that'll be used to plot the function.
#          For higher quality graphing, keep the number high but kindly note
#          that higher values for npoints significantly slows down the
#          animation; (default = 100)

time = 0
dt = 0.12
lower_xbound = -1
upper_xbound = 1
lower_ybound = -1
upper_ybound = 1
npoints = 100


###############################################################################
# xyz are the coordinates of the points that'll be used to plot the function
xyz = update_coordinates(lower_xbound, upper_xbound, lower_ybound, upper_ybound,
                 npoints, t=time)


###############################################################################
# colors and radius are used to manipulate the color and radius of the points
# respectively (default radius = 0.05)
# Ideally, radius should be kept somewhat low and npoints should be large.
colors = window.colors.yellow
radius = 0.05

###############################################################################
# Initializing point and axes actors and adding them to the scene.
point_actor = actor.point(xyz, colors, point_radius=radius)
axes_actor = actor.axes(scale=(1.5, 1.5, 1.5))
scene.add(axes_actor)
scene.add(point_actor)

###############################################################################
# Initializing text box to print the 2D function which is being rendered.
tb = ui.TextBlock2D(bold=True, position=(150, 60))
tb.message = "z = F(x, y, t) = 0.24*sin(x)*cos(y)*cos(t)"
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
end = 500

###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The 2D function is rendered here.
def timer_callback(_obj, _event):
    global xyz
    global time
    time += dt
    cnt = next(counter)
    xyz = update_coordinates(lower_xbound, upper_xbound, lower_ybound, upper_ybound,
                     npoints, t=time)
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