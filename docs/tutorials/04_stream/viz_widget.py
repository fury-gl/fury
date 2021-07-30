################################################
#
"""
=======================================================
Streaming FURY with WebRTC/MJPEG using the Widget Object
========================================================

Notes
------
For this example your python version should be 3.8 or greater

"""

import numpy as np
import time

from fury import actor, window
from fury.stream.widget import Widget

interactive = False
window_size = (720, 500)

centers = 1*np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
actors = actor.sphere(
        centers, opacity=.5, radii=.4, colors=colors)
scene = window.Scene()
scene.add(actors)
showm = window.ShowManager(
    scene,
    window_size[0], window_size[1])
widget = Widget(showm)

widget.start()
# open  your default  browser with the following url
# localhost:7777?encoding=mjpeg
time.sleep(5)
widget.stop()

if interactive:
    showm.start()

window.record(showm.scene, size=window_size, out_path="viz_widget.png")
