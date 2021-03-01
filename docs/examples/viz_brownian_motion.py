"""
======================================================================
Brownian motion
======================================================================
Brownian motion, or pedesis, is the random motion of particles suspended in
a medium. In this animation, path followed by 4 particles exhibiting brownian
motion in 3D is plotted.

Importing necessary modules
"""

from fury import window, actor, ui
import numpy as np
from scipy.stats import norm

###############################################################################
# initialize the random number generator
np.random.seed(24)

###############################################################################
# Variable(s) and their description-
# total_time: time to be discretized via time_steps (default: 5)
# num_total_steps: total number of steps each particle will take (default: 300)
# time_step = (total_time/num_total_steps): time step (default: 5/300)
# counter_step: to keep track of number of steps taken (initialised to 0)
# delta: delta determines the "speed" of the Brownian motion. Increase delta
#        to speed up the motion of the particle(s). The random variable
#        of the position has a normal distribution whose mean is the position
#        at counter_step=0 and whose variance is delta**2*time_step.
#        (default: 1.8)
# num_particles: number of particles whose path will be plotted (default: 10)
# coords: array which stores the coordinates of the particles
#         Initial origin, common to all particles: [0, 0, 0]
# colors: array which stores the colors of paths followed by the particles
#         (randomly generated colors for each particle)

total_time = 5
num_total_steps = 300
time_step = total_time/num_total_steps
counter_step = 0
delta = 1.8
num_particles = 10
coords = np.zeros((num_particles, 3))
colors = np.random.rand(num_particles, 3)

###############################################################################
# Function that generates the path of the particles


def generate_path(point, delta, time_step):
    x, y, z = point
    x += norm.rvs(scale=delta**2*time_step)
    y += norm.rvs(scale=delta**2*time_step)
    z += norm.rvs(scale=delta**2*time_step)
    return x, y, z


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
# Creating a container (cube actor) inside which the particle(s) move around

container_actor = actor.box(centers=np.array([[0, 0, 0]]),
                            colors=(0.7, 0.7, 1.0, 0.3), scales=6)
scene.add(container_actor)

###############################################################################
# Initializing text box to display the name of the animation

tb = ui.TextBlock2D(bold=True, position=(235, 40), color=(0, 0, 0))
tb.message = "Brownian Motion"
scene.add(tb)


###############################################################################
# The path of the particles exhibiting Brownian motion is plotted here.

def timer_callback(_obj, _event):
    global counter_step, num_particles, coords, delta, time_step, \
           num_total_steps
    counter_step += 1
    all_particles_pos = []

    # Plotting the path followed by the particle(s)
    for i in range(num_particles):
        coor_e = np.copy(coords[i])
        coords[i] = generate_path(coords[i], delta, time_step)
        all_particles_pos.append(np.array([coords[i], coor_e]))
    path_actor = actor.line(all_particles_pos, colors, linewidth=3)
    scene.add(path_actor)
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
