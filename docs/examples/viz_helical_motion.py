"""
======================================================================
Motion of a charged particle in a combined magnetic and electric field
======================================================================

A charged particle follows a curved path in a magnetic field.
In an electric field, the particle tends to accelerate in a direction
parallel/antiparallel to the electric field depending on the nature of
charge on the particle. In a combined electric and magnetic field,
the particle moves along a helical path.

In this animation, there's a magnetic and an electric field present in +x
direction under whose influence the positively charged particle follows a
helical path.

Importing necessary modules
"""

from fury import window, actor, utils, ui, colormap, material
import numpy as np


###############################################################################
# Let's define some variable and their description:
#
# * `radius_particle`: radius of the point that will represent the particle
#   (default = 0.08)
# * `initial_velocity`: initial velocity of the particle along +x
#   (default = 0.09)
# * `acc`: acceleration of the particle along +x (due to the electric field)
#   (default = 0.004)
# * `time`: time (default time i.e. time at beginning of the animation = 0)
# * `incre_time`: value by which time is incremented for each call of
#   timer_callback (default = 0.2)
# * `angular_frq`: angular frequency (default = 0.1)
# * `phase_angle`: phase angle (default = 0.002)
# * `radius_path`: radius of the path followed by the particle (default = 1.5)
#

radius_particle = 0.15
initial_velocity = 0.09
acc = 0.004
time = 0
incre_time = 0.2
angular_frq = 0.11
phase_angle = 0.002
radius_path = 1.5


###############################################################################
# Creating a scene object and configuring the camera's position

scene = window.Scene()
scene.background((0, 0.8, 0.5), gradient=True)
scene.zoom(1.3)
scene.set_camera(position=(13.499, 11.624, 15.546),
                 focal_point=(3.639, 0.559, 0.369),
                 view_up=(-0.271, 0.853, -0.446))
showm = window.ShowManager(scene, size=(800, 600), reset_camera=True)
showm.initialize()


###############################################################################
# Creating a blue colored arrow which shows the direction of magnetic field and
# electric field.

color_arrow = window.colors.red  # color of the arrow can be manipulated
centers = np.array([[0, 0, 0]])
directions = np.array([[1, 0, 0]])
heights = np.array([8])
arrow_actor = actor.arrow(centers, directions, color_arrow, heights,
                          resolution=100, tip_length=0.04375,
                          tip_radius=0.0125, shaft_radius=0.00375)

# manipulating shading for aesthetics
material.manifest_standard(arrow_actor, ambient_level=1)
axes_actor = actor.axes(scale=(0, 1, 1))
material.manifest_standard(axes_actor, ambient_level=1)
scene.add(arrow_actor, axes_actor)

###############################################################################
# Initializing the initial coordinates of the particle

x = initial_velocity*time + 0.5*acc*(time**2)
y = radius_path*np.sin(angular_frq*time + phase_angle)
z = radius_path*np.cos(angular_frq*time + phase_angle)
pts = np.array([[x, y, z]])


###############################################################################
# Initializing the path_actor which will trace the path followed by the
# charged particle

num_total_steps = 225  # total number of simulation steps
origin = np.asarray([0, 0, 0], dtype=float)
position = np.tile(origin, (num_total_steps, 1))

v = np.linspace(0, 1, num_total_steps)
cmap_name = 'gist_heat'  # using colormap to color the path of the particle
colors = colormap.create_colormap(v, name=cmap_name)

path_actor = actor.line([position], colors, linewidth=3)
actor.attributes_to_actor(path_actor, position=position,
                          no_vertices_per_point_divisor=num_total_steps)
scene.add(path_actor)

###############################################################################
# Initializing point actor which will represent the charged particle

color_particle = window.colors.yellow  # color of particle can be manipulated
charge_actor = actor.point(pts, color_particle, point_radius=radius_particle,
                           phi=32, theta=32)
actor.attributes_to_actor(charge_actor, pts)

# manipulating shading for aesthetics
material.manifest_standard(charge_actor, ambient_level=1)
scene.add(charge_actor)

###############################################################################
# Initializing text box to display the name of the animation

tb = ui.TextBlock2D(bold=True, position=(40, 60), color=(0, 0, 0),
                    font_size=20)
m1 = "Motion of a charged particle in a "
m2 = "combined electric and magnetic field"
tb.message = m1 + m2
scene.add(tb)

###############################################################################
# Initializing text box to display the velocity of the particle
tb2 = ui.TextBlock2D(text="Velocity of the particle\n" +
                     "  along x-axis = \n" +
                     "  along y-axis = \n" +
                     "  along z-axis = ", position=(50, 500),
                     font_size=15, color=(1, 1, 1), bold=True)
scene.add(tb2)

###############################################################################
# Initializing counter
counter = 0


###############################################################################
# Coordinates to be plotted are changed everytime timer_callback is called by
# using the update_coordinates function. The wave is rendered here.
def timer_callback(_obj, _event):
    global time, counter
    time += incre_time

    # coordinates of the particle
    x = initial_velocity*time + 0.5*acc*(time**2)
    y = radius_path*np.sin(10*angular_frq*time + phase_angle)
    z = radius_path*np.cos(10*angular_frq*time + phase_angle)
    pts = np.array([[x, y, z]])

    # computing and displaying the velocity of the particle along x, y, z axes
    vx = initial_velocity + acc*time
    vy = 10*angular_frq*radius_path*np.cos(10*angular_frq*time + phase_angle)
    vz = -10*angular_frq*radius_path*np.sin(10*angular_frq*time + phase_angle)

    if(counter % 3 == 0):
        tb2.message = "Velocity of the particle\n" + \
                      "  along x axis = {:.2f}\n".format(vx) + \
                      "  along y axis = {:.2f}\n".format(vy) + \
                      "  along z axis = {:.2f}".format(vz)

    # updating the charged particle
    charge_actor.vertices[:] = charge_actor.initial_vertices + \
        np.repeat(pts, charge_actor.no_vertices_per_point, axis=0)
    utils.update_actor(charge_actor)

    # updating the path traced
    counter += 1
    path_actor.position[counter:] = [x, y, z]
    path_actor.vertices[:] = path_actor.initial_vertices + \
        np.repeat(path_actor.position, path_actor.no_vertices_per_point,
                  axis=0)
    utils.update_actor(path_actor)

    showm.render()

    # to end the animation
    if counter == num_total_steps-1:
        showm.exit()


###############################################################################
# Run every 15 milliseconds
showm.add_timer_callback(True, 15, timer_callback)
showm.start()
window.record(showm.scene, size=(800, 600), out_path="viz_helical_motion.png")
