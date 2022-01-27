"""
=======================================================
Multithreading Example
=======================================================

The goal of this demo is to show how to use different threads
to interact with fury. In particular, the main thread is used
to update interactions and render the scene, while thread A
rotates the camera, thread B prints a counter, and thread C
adds and removes elements from the scene.
"""

from fury import window, actor, ui
from threading import Thread
import numpy as np
from fury.lib import Command
import time


# Preparing to draw some spheres
xyz = 10 * (np.random.rand(100, 3)-0.5)
colors = np.random.rand(100, 4)
radii = np.random.rand(100) + 0.5

scene = window.Scene()
sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)
scene.add(sphere_actor)


# Preparing the show manager as usual
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

# Creating a text block to show a message and reset the camera
tb = ui.TextBlock2D(bold=True)
scene.add(tb)
scene.ResetCamera()


# Create a function to print a counter to the console
def print_counter():
    counter = 0
    print("")
    for _ in range(10000):
        print("\rCounter: %d" % counter, end="")
        counter += 1
        time.sleep(1)
        if(showm.is_done()):
            break
    print("")

# Create a function to rotate the camera


def rotate_camera():
    for i in range(1000):
        message = "Let's count up to 100 and exit :" + str(i+1)
        if(showm.lock_current()):
            tb.message = message
            scene.azimuth(0.01 * i)
            showm.release_current()
            time.sleep(0.01)
        else:
            break

# Create a function to add or remove the axes and increase its scale


def add_remove_axes():
    current_axes = None
    for i in range(1000):
        if(showm.lock_current()):
            if(current_axes is None):
                current_axes = actor.axes(scale=(0.05 * i, 0.05 * i, 0.05 * i))
                scene.add(current_axes)
            else:
                scene.rm(current_axes)
                current_axes = None
            showm.release_current()
            time.sleep(0.5)
        else:
            break


##############################################################################
# Start the threads
# Multiple threads can be started here
# First, one to rotate the camera
thread_a = Thread(target=rotate_camera)
thread_a.start()

# Now let's start a thread that will print a counter
thread_b = Thread(target=print_counter)
thread_b.start()

# Now let's start a thread that will add or remove axes
thread_c = Thread(target=add_remove_axes)
thread_c.start()

# Let's start the show manager loop with multithreading option
showm.start(multithreaded=True)

# Wait for the threads to finish
thread_a.join()
thread_b.join()
thread_c.join()
