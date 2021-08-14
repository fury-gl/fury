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

import asyncio
import numpy as np

from fury import actor, window
from fury.stream.widget import Widget

interactive = False
window_size = (720, 500)

centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
actors = actor.sphere(
        centers, opacity=.5, radii=.4, colors=colors)
scene = window.Scene()
scene.add(actors)
showm = window.ShowManager(
    scene,
    size=(window_size[0], window_size[1]))
widget = Widget(showm, port=8000)

time_sleep = 1000 if interactive else 1


async def main():
    widget.start()
    await asyncio.sleep(time_sleep)
    widget.stop()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

window.record(showm.scene, size=window_size, out_path="viz_widget.png")
