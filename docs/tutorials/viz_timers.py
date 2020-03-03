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


import numpy as np
from fury import window, actor, ui, transform
import itertools

xyz = 10 * np.random.rand(100, 3)
directions = xyz.copy() / 10.

colors = np.random.rand(100, 3)
radii = np.random.rand(100) + 0.5

scene = window.Scene()

sphere_actor = actor.arrow(centers=xyz,
                           directions=directions,
                           colors=colors)

# sphere_actor = actor.sphere(centers=xyz,
#                             colors=colors)

scene.add(sphere_actor)

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=False)

showm.initialize()

tb = ui.TextBlock2D(bold=True)

# use itertools to avoid global variables
counter = itertools.count()

pts = transform.vertices(sphere_actor, False)

print(pts.shape)


sphere_actor.GetProperty().BackfaceCullingOn()
#sphere_actor.GetProperty().FrontfaceCullingOff()

displacements = np.zeros((pts.shape[0], 3), 'f4')

sphere_update = transform.affine(sphere_actor, displacements)


def timer_callback(_obj, _event):
    cnt = next(counter)
    tb.message = "Let's count up to 100 and exit :" + str(cnt)
    global displacements
    # displacements += 0.05 * (np.random.rand(*displacements.shape) - 0.5)
    displacements += np.array([.5, 0, 0], dtype='f4')
    sphere_update.update()
    # sphere_actor.GetMapper().GetInput().ComputeBounds()
    # showm.scene.azimuth(0.05 * cnt)
    # sphere_actor.GetProperty().SetOpacity(cnt/100.)
    showm.render()
    if cnt == 100:
        showm.exit()


scene.add(tb)

# Run every 200 milliseconds
showm.add_timer_callback(True, 200, timer_callback)

showm.start()

window.record(showm.scene, size=(900, 768), out_path="viz_timer.png")
