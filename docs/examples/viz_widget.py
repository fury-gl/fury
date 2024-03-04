"""
========================================================
Streaming FURY with WebRTC/MJPEG using the Widget Object
========================================================

The Widget Object simplifies the process of streaming
your data visualization with WebRTC or MJPEG. Encoding.
Other users can view and interact with your data visualization
in real-time using a web-browser.

By default, the widget will open a local server on port 8000.
With the encoding parameter you can choose between mjpeg or
webrtc. WebRTC is a more robust option and can be used to perform
a live streaming with a low-latency connection. However, to use
webRTC you need to install the aiortc library.

.. code-block:: bash

    pip install aiortc

In addition, if you don't have ffmpeg installed, you  need
to install it.

Linux


`apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config`

OS X

`brew install ffmpeg opus libvpx pkg-config`

Notes
-----
For this example your python version should be 3.8 or greater

"""

import asyncio
import platform
import time

import numpy as np

from fury import actor, window
from fury.stream.widget import Widget

interactive = False
window_size = (720, 500)
N = 4
centers = np.random.normal(size=(N, 3))
colors = np.random.uniform(0.1, 1.0, size=(N, 3))
actors = actor.sphere(centers, opacity=0.5, radii=0.4, colors=colors)
scene = window.Scene()
scene.add(actors)
showm = window.ShowManager(scene, size=(window_size[0], window_size[1]))

##########################################################################
# Create a stream widget

widget = Widget(showm, port=8000)

# if you want to use webRTC, you can pass the argument to choose this encoding
# which is a more robust option.
# `widget = Widget(showm, port=8000, encoding='webrtc')`

time_sleep = 1000 if interactive else 1

###########################################################################
# If you want to use the widget in a Windows environment without the WSL
# you need to use the asyncio version of the widget.
#
if platform.system() == 'Windows':

    async def main():
        widget.start()
        await asyncio.sleep(time_sleep)
        widget.stop()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
else:
    # Running the widget in a normal Python environment in Linux or MacOS
    # we can use the normal version of the widget.
    widget.start()
    time.sleep(time_sleep)
    widget.stop()

window.record(showm.scene, size=window_size, out_path='viz_widget.png')
