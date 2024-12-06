"""
===============
Using a timer
===============

This example shows how to create a simple animation using a timer callback.

We will use a sphere actor that generates many spheres of different colors,
radii and opacity. Then we will animate this actor by rotating and changing
global opacity levels from inside a user defined callback.

The timer will call this user defined callback every 200 milliseconds. The
application will exit after the callback has been called 100 times.
"""

import itertools

import numpy as np

import fury

xyz = 10 * np.random.rand(100, 3)
colors = np.random.rand(100, 4)
radii = np.random.rand(100) + 0.5

scene = fury.window.Scene()

sphere_actor = fury.actor.sphere(centers=xyz, colors=colors, radii=radii)

scene.add(sphere_actor)

showm = fury.window.ShowManager(
    scene=scene, size=(900, 768), reset_camera=False, order_transparent=True
)


tb = fury.ui.TextBlock2D(bold=True)

# use itertools to avoid global variables
counter = itertools.count()


def timer_callback(_obj, _event):
    global timer_id
    cnt = next(counter)
    tb.message = "Let's count up to 300 and exit :" + str(cnt)
    showm.scene.azimuth(0.05 * cnt)
    sphere_actor.GetProperty().SetOpacity(cnt / 100.0)
    showm.render()

    if cnt == 10:
        # destroy the first timer and replace it with another faster timer
        showm.destroy_timer(timer_id)
        timer_id = showm.add_timer_callback(True, 10, timer_callback)

    if cnt == 300:
        # destroy the second timer and exit
        showm.destroy_timer(timer_id)
        showm.exit()


scene.add(tb)

# Run every 200 milliseconds
timer_id = showm.add_timer_callback(True, 200, timer_callback)

showm.start()

fury.window.record(scene=showm.scene, size=(900, 768), out_path="viz_timer.png")
