"""
===============================================
Animating a time-varying 2D wave function
===============================================

This is a simple demonstration of how one can visualize
time-varying 2D wave functions using FURY.
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


def update_coordinates(x, y, equation, time=0, cmap_name='viridis'):

    # z is the function F i.e. F(x, y, t)
    z = eval(equation)
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
#          (default = 128)
#

time = 0
dt = 0.1
lower_xbound = -1
upper_xbound = 1
lower_ybound = -1
upper_ybound = 1
npoints = 128

###############################################################################
# creating the x, y points which will be used to fit the equation to get
# elevation and generate the surface
x = np.linspace(lower_xbound, upper_xbound, npoints)
y = np.linspace(lower_ybound, upper_ybound, npoints)
x, y = np.meshgrid(x, y)
x = x.reshape(-1)
y = y.reshape(-1)

###############################################################################
# Initializing wave_actors and adjusting lighting options to make the waves
# look more aesthetically pleasant.


def wave(x, y, equation, colormap_name):
    xyz, colors = update_coordinates(x, y, equation=equation, time=time,
                                     cmap_name=colormap_name)
    wave_actor = actor.surface(xyz, colors=colors)
    wave_actor.equation = equation
    wave_actor.cmap_name = colormap_name
    wave_actor.GetProperty().SetAmbient(1)
    wave_actor.vertices = utils.vertices_from_actor(wave_actor)
    wave_actor.no_vertices_per_point = len(wave_actor.vertices)/npoints**2
    wave_actor.initial_vertices = wave_actor.vertices.copy() - \
        np.repeat(xyz, wave_actor.no_vertices_per_point, axis=0)
    return wave_actor


###############################################################################
# Equations to be plotted
eq1 = "(np.sin(np.cos(3*np.pi*y)-time)*x+y*np.sin(np.cos(x*3*np.pi)-time))*\
      0.48"
eq2 = "0.72*((x**2 - y**2)/(x**2 + y**2))**(2)*np.cos(6*np.pi*x*y-time)/3"
eq3 = "(np.sin(np.pi*2*x-np.sin(1.2*time))*np.cos(np.pi*2*y+np.cos(1.2*time)))\
      *0.48"
eq4 = "np.sin(2*np.pi*x**2+time)*np.cos(2*np.pi*y**2-time)*0.48"
equations = [eq1, eq2, eq3, eq4]

###############################################################################
# List of colormaps to be used for the various functions.
cmap_names = ['YlOrRd', 'plasma', 'viridis', 'Blues']

###############################################################################
# Creating a list of wave actors.
waves = []
for i in range(4):
    waves.append(wave(x, y, equation=equations[i],
                 colormap_name=cmap_names[i]))
actors = waves


###############################################################################
# Creating a scene object and configuring the camera's position

scene = window.Scene()
scene.set_camera(position=(4.5, -15, 20), focal_point=(4.5, 0.0, 0.0),
                 view_up=(0.0, 0.0, 1.0))
showm = window.ShowManager(scene, size=(600, 600))

###############################################################################
# Creating an axes actor(for reference) and a grid to store multiple functions.
# Adding them both to the scene.

# To store the function names
text = []
for i in range(4):
    t_actor = actor.text_3d("Function " + str(i+1), position=(-10, 0, 0),
                            font_size=0.3, justification='center')
    text.append(t_actor)

# grid
grid_ui = ui.GridUI(actors=actors, captions=text,
                    caption_offset=(0, -2, 0), dim=(1, 4),
                    cell_padding=2,
                    aspect_ratio=1,
                    rotation_axis=(0, 1, 0))
showm.scene.add(grid_ui)
showm.scene.add(actor.axes())


###############################################################################
# Initializing text box to print the title of the animation
tb = ui.TextBlock2D(bold=True, position=(200, 60))
tb.message = "Animated 2D functions"
scene.add(tb)

###############################################################################
# Initializing showm and counter
showm.initialize()
counter = itertools.count()

###############################################################################
# end is used to decide when to end the animation
end = 200


###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The 2D functions are rendered here

def timer_callback(_obj, _event):
    global xyz, time
    time += dt
    cnt = next(counter)

    # updating the colors and vertices of the triangles used to form the waves
    for wave in waves:
        xyz, colors = update_coordinates(x, y, equation=wave.equation,
                                         time=time, cmap_name=wave.cmap_name)
        wave.GetMapper().GetInput().GetPointData().\
            SetScalars(utils.numpy_to_vtk_colors(255*colors))
        wave.vertices[:] = wave.initial_vertices + \
            np.repeat(xyz, wave.no_vertices_per_point, axis=0)
        utils.update_actor(wave)

    showm.render()
    # to end the animation
    if cnt == end:
        showm.exit()


###############################################################################
# Run every 30 milliseconds
showm.add_timer_callback(True, 30, timer_callback)

interactive = True
if interactive:
    showm.start()

window.record(showm.scene, size=(600, 600), out_path="2Dwave_func.png")
