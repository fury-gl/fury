"""
===============================================
Animating a time-varying 2D wave function
===============================================

This is a simple demonstration of how one can visualize
time-varying 2D wave functions using FURY.

Some other interesting functions you can try plotting -
#z = (x**2 - y**2)/(x**2 + y**2)**0.5*np.cos(np.pi*x-rate*time)
#z = np.sin(x**2)*np.cos(y**2-rate*incre)#*np.sin(rate*time)
#z = np.sin(x**2-y**2)*np.sin(time)

"""

###############################################################################
# Importing necessary modules

from fury import window, actor, ui, utils, colormap
import numpy as np
import itertools


###############################################################################
# The following function is used to create and/or update the coordinates of the
# points which are being used to plot the wave.
# Kindly note that only the z coordinate is being modified with time as only
# the z coordinate is a function of time.

def update_coordinates(time=0, incre=0, cmap_name='viridis'):

    # x, y points which will be used to fit the equation to get elevation and
    # generate the surface.
    global x, y

    # manipulate rate to change the speed of the waves; (default = 1)
    rate = 1

    # Z is the function F i.e. F(x, y, t)
    z = 0.48*np.sin(2*np.pi*x)*np.cos(2*np.pi*y-rate*time)
    xyz = np.vstack([x, y, z]).T

    # creating the colormap
    v = np.copy(z)
    v /= np.max(np.abs(v), axis=0)
    colors = colormap.create_colormap(v, name=cmap_name)

    return xyz, colors


###############################################################################
# Variables and their usage-
# time: float
#       initial value of the time variable i.e. value of the time variable at
#       the beginning of the program; (default = 0)
# dt: float
#     amount by which the "time" variable is incremented for every iteration
#     of timer_callback function; (default = 0.1)
# lower_xbound: float
#               lower bound of the x values in which the function is plotted
#               (default = -1)
# upper_xbound: float
#               Upper bound of the x values in which the function is plotted
#               (default = 1)
# lower_ybound: float
#               lower bound of the y values in which the function is plotted
#               (default = -1)
# upper_ybound: float
#               Upper bound of the y values in which the function is plotted
#               (default = 1)
# npoints: int
#          For high quality rendering, keep the number high but kindly note
#          that higher values for npoints slows down the animation
#          (default = 300)
# cmap_name: string
#            name of the colormap being used to color the wave
#            (default = 'plasma')

time = 0
dt = 0.1
lower_xbound = -1
upper_xbound = 1
lower_ybound = -1
upper_ybound = 1
npoints = 300
cmap_name = 'plasma'

###############################################################################
# creating the x, y points which will be used to fit the equation to get
# elevation and generate the surface
x = np.linspace(lower_xbound, upper_xbound, npoints)
y = np.linspace(lower_ybound, upper_ybound, npoints)
x, y = np.meshgrid(x, y)
x = x.reshape(-1)
y = y.reshape(-1)

###############################################################################
# xyz are the coordinates of the points that'll be used to plot the wave
# colors refers to the colormap that'll be used to color the wave
xyz, colors = update_coordinates(time=time, cmap_name=cmap_name)

###############################################################################
# Initializing wave_actor and adjusting lighting options to make the wave look
# more aesthetically pleasant
wave_actor = actor.surface(xyz, colors=colors)
wave_actor.GetProperty().SetAmbient(1)

###############################################################################
# Creating a scene object and configuring the camera's position

scene = window.Scene()
scene.zoom(12.5)
scene.set_camera(position=(60, 40, 30), focal_point=(0.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 1.0))
showm = window.ShowManager(scene, size=(600, 600))

###############################################################################
# Adding the wave_actor to the scene
scene.add(wave_actor)


###############################################################################
# Initializing text box to print the formula of the 2D function which is being
# animated
tb = ui.TextBlock2D(bold=True, position=(80, 60))
tb.message = "z = F(x, y, t) = 0.48*sin(2*pi*x)*cos(2*pi*y-rate*t)"
scene.add(tb)

###############################################################################
# Initializing showm and counter
showm.initialize()
counter = itertools.count()

###############################################################################
# vertices stores the coordinates and of the triangles used to form the
# wave_actor, vertices are updated with time as the wave changes its shape

vertices = utils.vertices_from_actor(wave_actor)
no_vertices_per_point = len(vertices)/npoints**2
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_point, axis=0)

###############################################################################
# end is used to decide when to end the animation
end = 200

###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The 2D function is rendered here


def timer_callback(_obj, _event):
    global xyz, time
    time += dt
    cnt = next(counter)

    # updating the colors and vertices of the triangles used to form the wave
    xyz, colors = update_coordinates(time=time, incre=cnt, cmap_name=cmap_name)

    # updating colors
    wave_alias = wave_actor.GetMapper().GetInput().GetPointData()
    wave_alias.SetScalars(utils.numpy_to_vtk_colors(255*colors))

    # updating the vertices
    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_point, axis=0)
    utils.update_actor(wave_actor)
    showm.render()

    # to end the animation
    if cnt == end:
        showm.exit()

###############################################################################
# Run every 30 milliseconds


showm.add_timer_callback(True, 30, timer_callback)

interactive = False
if interactive:
    showm.start()

window.record(showm.scene, size=(600, 600), out_path="2Dwave_func.png")
