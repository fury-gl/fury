"""
=======================================================
Multithreading Example
=======================================================

The goal of this demo is to show how to use different threads
to interact with FURY. In particular, the main thread is used
to update interactions and render the scene, while thread A
rotates the camera, thread B prints a counter, and thread C
adds and removes elements from the scene.
"""

from threading import Thread
import time

import numpy as np

from fury import actor, ui, window

# Preparing to draw some spheres
xyz = 10 * (np.random.random((100, 3)) - 0.5)
colors = np.random.random((100, 4))
radii = np.random.random(100) + 0.5

scene = window.Scene()
sphere_actor = actor.sphere(
    centers=xyz, colors=colors, radii=radii, use_primitive=False
)
scene.add(sphere_actor)


# Preparing the show manager as usual
showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)

# showm.initialize()

# Creating a text block to show a message and reset the camera
tb = ui.TextBlock2D(bold=True)
scene.add(tb)
scene.ResetCamera()


# Create a function to print a counter to the console
def print_counter():
    print('')
    for i in range(100):
        print('\rCounter: %d' % i, end='')
        message = "Let's count up to 100 and exit :" + str(i + 1)
        tb.message = message
        time.sleep(0.05)
        if showm.is_done():
            break
    showm.exit()
    print('')


# Create a function to rotate the camera


def rotate_camera():
    for i in range(100):
        if showm.lock_current():
            scene.azimuth(0.01 * i)
            showm.release_current()
            time.sleep(0.05)
        else:
            break


# Create a function to add or remove the axes and increase its scale


def add_remove_axes():
    current_axes = None
    for i in range(100):
        if showm.lock_current():
            if current_axes is None:
                current_axes = actor.axes(scale=(0.20 * i, 0.20 * i, 0.20 * i))
                scene.add(current_axes)
                pass
            else:
                scene.rm(current_axes)
                current_axes = None
                pass
            showm.release_current()
            time.sleep(0.1)
        else:
            break


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
