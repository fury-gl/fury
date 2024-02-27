"""
====================================
Streaming FURY with user interaction
====================================

In this tutorial, we show how to use the FURY Streaming system to
serve an interactive visualization through a web browser.

You can choose between two different encodings: WebRTC or MJPEG.
WebRTC is a more robust option and can be used to perform
a live streaming with a low-latency connection for example using
ngrok. However, to use webRTC you need to install the aiortc library.

.. code-block:: bash

    pip install aiortc


Notes
-----
If you don't have ffmpeg installed, you need to install it to use WebRTC

Linux


`apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config`

OS X

`brew install ffmpeg opus libvpx pkg-config`

"""

import multiprocessing
import sys

import numpy as np

from fury import actor, window
from fury.stream.client import FuryStreamClient, FuryStreamInteraction

# if this example it's not working for you and you're using MacOs
# uncomment the following line
# multiprocessing.set_start_method('spawn')
from fury.stream.server.main import WEBRTC_AVAILABLE, web_server, web_server_raw_array

if __name__ == '__main__':
    interactive = False
    # `use_raw_array` is a flag to tell the server to use python RawArray
    # instead of SharedMemory which is a new feature in python 3.8
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Array
    # https://docs.python.org/3/library/multiprocessing.html#shared-memory-objects

    use_raw_array = sys.version_info < (3, 8)
    window_size = (300, 300)
    # `max_window_size` are the maximum size of the window that will be
    # allowed to be sent to the browser. For example, if you set
    # `max_window_size=(800, 800)` then the browser will be limited to
    # a window of size (800, 800).

    max_window_size = (400, 400)
    # 0 ms_stream means that the frame will be sent to the server
    # right after the rendering

    # `ms_interaction` is the time in milliseconds that the user will have
    # to interact with the visualization

    ms_interaction = 1
    # `ms_stream` is the number of milliseconds that the server will
    # wait before sending a new frame to the browser. If `ms_stream=0`
    # then the server will send the frame right after the rendering.

    ms_stream = 0
    # max number of interactions to be stored inside the queue
    max_queue_size = 17
    ######################################################################
    centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    actors = actor.sphere(centers, opacity=0.5, radii=0.4, colors=colors)
    scene = window.Scene()

    scene.add(actors)

    showm = window.ShowManager(scene, size=(window_size[0], window_size[1]))

    stream = FuryStreamClient(
        showm, max_window_size=max_window_size, use_raw_array=use_raw_array
    )
    stream_interaction = FuryStreamInteraction(
        showm, max_queue_size=max_queue_size, use_raw_array=use_raw_array
    )

    if use_raw_array:
        p = multiprocessing.Process(
            target=web_server_raw_array,
            args=(
                stream.img_manager.image_buffers,
                stream.img_manager.info_buffer,
                stream_interaction.circular_queue.head_tail_buffer,
                stream_interaction.circular_queue.buffer._buffer,
                8000,
                'localhost',
                True,
                WEBRTC_AVAILABLE,
            ),
        )

    else:
        p = multiprocessing.Process(
            target=web_server,
            args=(
                stream.img_manager.image_buffer_names,
                stream.img_manager.info_buffer_name,
                stream_interaction.circular_queue.head_tail_buffer_name,
                stream_interaction.circular_queue.buffer.buffer_name,
                8000,
                'localhost',
                True,
                WEBRTC_AVAILABLE,
            ),
        )
    p.start()
    stream_interaction.start(ms=ms_interaction)
    stream.start(
        ms_stream,
    )
    ###########################################################################
    # If you have aiortc in your system, you can see your live streaming
    # through the following url: htttp://localhost:8000/?encoding=webrtc
    # Other wise, you can use the following url:
    # http://localhost:8000/?encoding=mjpeg

    if interactive:
        showm.start()

    # We need to close the server after the show is over
    p.kill()
    ###########################################################################
    # We release the resources and stop the interactive mode
    stream.stop()
    stream_interaction.stop()
    stream.cleanup()
    stream_interaction.cleanup()

    window.record(showm.scene, size=window_size, out_path='viz_interaction.png')
