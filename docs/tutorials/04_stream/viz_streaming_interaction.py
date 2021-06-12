from os.path import join as pjoin
from fury import actor, window, colormap as cmap
import numpy as np
import time

import multiprocessing
from fury.stream.servers.webrtc.server import webrtc_server
from fury.stream.client import FuryStreamClient, FuryStreamInteraction

import logging
import time


# note, if python version is equal or higher than 3.8
# uses shared memory approach
if __name__ == '__main__':
    use_high_res = False
    if use_high_res:
        window_size = (1280, 720)
        max_window_size = (1920, 1080)
    else:
        window_size = (720, 500)
        max_window_size = (600, 600)
    # 0 ms_stream means that the frame will be sent to the server
    # right after the rendering
    ms_interaction = 10
    ms_stream = 16
    # max number of interactions to be stored inside the queue
    max_queue_size =  17 
    ######################################################################
    centers = 1*np.array([
        [0, 0, 0],
        [-1, 0, 0],
        [1, 0, 0]
    ])
    centers2 = centers - np.array([[0, -1, 0]])
    # centers_additive = centers_no_depth_test - np.array([[0, -1, 0]])
    # centers_no_depth_test2 = centers_additive - np.array([[0, -1, 0]])
    colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    actors = actor.sdf(
        centers, primitives='sphere', colors=colors, scales=2)

    actors2 = actor.sphere(
        centers2, opacity=.5, radii=.4, colors=colors)
    scene = window.Scene()

    # scene.add(lines_actor)
    scene.add(actors)
    scene.add(actors2)

    interactive = False

    # scene.set_camera(
    #     position=(0, 0, 1000), focal_point=(0.0, 0.0, 0.0),
    #     view_up=(0.0, 0.0, 0.0))

    showm = window.ShowManager(scene, reset_camera=False, size=(
        window_size[0], window_size[1]), order_transparent=False,
        # multi_samples=8
    )


    ##############################################################################
    # ms define the amount of mileseconds that will be used in the timer event.
    # Otherwise, if ms it's equal to zero the shared memory it's updated in each 
    # render event
    # showm.window.SetOffScreenRendering(1)
    # showm.window.EnableRenderOff()
    showm.initialize()

    stream = FuryStreamClient(
        showm, window_size, max_window_size=max_window_size,)
    stream_interaction = FuryStreamInteraction(
        showm, max_queue_size=max_queue_size, fury_client=stream)
    # linux
    # p = multiprocessing.Process(
    #     target=webrtc_server,
    #     args=(stream, None, None, circular_queue))
    # osx,
    
    p = multiprocessing.Process(
        target=webrtc_server,
        args=(
            None, 
            stream.image_buffers,
            stream.image_buffer_names,
            stream.info_buffer,
            stream.info_buffer_name,
            None,
            stream_interaction.circular_queue.head_tail_buffer,
            stream_interaction.circular_queue.buffer._buffer,
            None,
            None,
            #stream_interaction.circular_queue.head_tail_buffer_name,
            #stream_interaction.circular_queue.buffer.buffer_name
            )
            )
    p.start()
    stream_interaction.start(ms=ms_interaction)
    stream.init(ms_stream,)
    showm.start()
    p.kill()
    stream.cleanup()

    # open a browser using the following the url
    # http://localhost:8000/
