################################################
# For this example your python version should be 3.8 or greater
#

import numpy as np
import time

from fury import actor, window
from fury.stream.widget import Widget

window_size = (720, 500)
centers = 1*np.array([[0, 0, 0],[-1, 0, 0],[1, 0, 0]])
colors = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
actors = actor.sdf(
    centers, primitives='torus', colors=colors, scales=2)

scene = window.Scene()
scene.add(actors)
interactive = False
showm = window.ShowManager(scene, reset_camera=False, size=(
    window_size[0], window_size[1]), order_transparent=False,
    # multi_samples=8
)
widget = Widget(showm, port=7777)

widget.start()
# open  your default  browser with the following url
# localhost:7777?encoding=mjpeg
time.sleep(100)
widget.stop()
