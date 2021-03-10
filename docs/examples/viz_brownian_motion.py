"""
======================================================================
Brownian motion
======================================================================
Brownian motion, or pedesis, is the random motion of particles suspended in
a medium. In this animation, path followed by 20 particles exhibiting brownian
motion in 3D is plotted.

Importing necessary modules
"""

from fury import window, actor, ui, utils
import numpy as np
from scipy.stats import norm

###############################################################################
# Variable(s) and their description-
# total_time: time to be discretized via time_steps (default: 5)
# num_total_steps: total number of steps each particle will take (default: 300)
# time_step = (total_time / num_total_steps)
#             time_step is computed and initialised in the __init__ method of
#             the particle class
# counter_step: to keep track of number of steps taken (initialised to 0)
# delta: delta determines the "speed" of the Brownian motion. Increase delta
#        to speed up the motion of the particle(s). The random variable
#        of the position has a normal distribution whose mean is the position
#        at counter_step = 0 and whose variance is equal to delta**2*time_step.
#        (default: 1.8)
# num_particles: number of particles whose path will be plotted (default: 20)
# path_thickness: thickness of line(s) that will be used to plot the path(s)
#                 of the particle(s) (default: 3)
# origin: coordinate from which the the particle(s) begin the motion
#         (default: [0, 0, 0])

total_time = 5
num_total_steps = 300
counter_step = 0
delta = 1.8
num_particles = 20
path_thickness = 3
origin = [0, 0, 0]

###############################################################################
# class particle is used to store and update coordinates of the particles (the
# path of the particles)


class particle:
    def __init__(self, colors, origin=[0, 0, 0], num_total_steps=300,
                 total_time=5, delta=1.8, path_thickness=3):
        origin = np.asarray(origin, dtype=float)
        self.position = np.tile(origin, (num_total_steps, 1))
        self.colors = colors
        self.delta = delta
        self.num_total_steps = num_total_steps
        self.time_step = total_time / num_total_steps
        self.path_actor = actor.line([self.position], colors,
                                     linewidth=path_thickness)
        self.vertices = utils.vertices_from_actor(self.path_actor)
        self.vcolors = utils.colors_from_actor(self.path_actor, 'colors')
        self.no_vertices_per_point = len(self.vertices) / num_total_steps
        nvpp = self.no_vertices_per_point
        self.initial_vertices = self.vertices.copy() - np.repeat(self.position,
                                                                 nvpp, axis=0)

    def update_path(self, counter_step):
        if counter_step < self.num_total_steps:
            x, y, z = self.position[counter_step-1]
            x += norm.rvs(scale=self.delta**2 * self.time_step)
            y += norm.rvs(scale=self.delta**2 * self.time_step)
            z += norm.rvs(scale=self.delta**2 * self.time_step)
            self.position[counter_step:] = [x, y, z]
            self.vertices[:] = self.initial_vertices + \
                np.repeat(self.position, self.no_vertices_per_point, axis=0)
            utils.update_actor(self.path_actor)


###############################################################################
# Creating a scene object and configuring the camera's position

scene = window.Scene()
scene.background((1.0, 1.0, 1.0))
scene.zoom(1.7)
scene.set_camera(position=(0, 0, 40), focal_point=(0.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 0.0))
showm = window.ShowManager(scene,
                           size=(600, 600), reset_camera=True,
                           order_transparent=True)
showm.initialize()

###############################################################################
# Creating a list of particle objects

list_particles = []
for i in range(num_particles):
    _particle = particle(colors=np.random.rand(1, 3), origin=origin,
                         num_total_steps=num_total_steps,
                         total_time=total_time, path_thickness=path_thickness)
    list_particles.append(_particle)
    scene.add(list_particles[i].path_actor)

###############################################################################
# Creating a container (cube actor) inside which the particle(s) move around

container_actor = actor.box(centers=np.array([[0, 0, 0]]),
                            colors=(0.5, 0.9, 0.7, 0.4), scales=6)
scene.add(container_actor)

###############################################################################
# Initializing text box to display the name of the animation

tb = ui.TextBlock2D(bold=True, position=(235, 40), color=(0, 0, 0))
tb.message = "Brownian Motion"
scene.add(tb)


###############################################################################
# The path of the particles exhibiting Brownian motion is plotted here

def timer_callback(_obj, _event):
    global counter_step, list_particles
    counter_step += 1
    for _particle in list_particles:
        _particle.update_path(counter_step=counter_step)
    showm.render()
    scene.azimuth(2)
    if (counter_step == num_total_steps):
        showm.exit()

###############################################################################
# Run every 30 milliseconds


showm.add_timer_callback(True, 30, timer_callback)
interactive = False
if interactive:
    showm.start()

window.record(showm.scene, size=(600, 600), out_path="viz_brownian_motion.png")
