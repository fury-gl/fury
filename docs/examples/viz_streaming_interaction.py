from os.path import join as pjoin
from fury import actor, window, colormap as cmap
import numpy as np

###############################################################################
# Then let's download some available datasets.


import multiprocessing
from fury.stream.servers.webrtc.server import webrtc_server

if __name__ == '__main__':


    ##############################################################################
    # First we will set the resolution which it'll be used by the streamer
    ms_interaction = 1
    window_size = (400, 400)
    ms_stream = 16
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

    from fury.stream.client import FuryStreamClient, FuryStreamInteraction

    ##############################################################################
    # ms define the amount of mileseconds that will be used in the timer event.
    # Otherwise, if ms it's equal to zero the shared memory it's updated in each 
    # render event

    stream_interaction = FuryStreamInteraction(showm, max_queue_size=50)
    showm.initialize()

    stream = FuryStreamClient(
        showm, window_size,)
    # linux
    # p = multiprocessing.Process(
    #     target=webrtc_server,
    #     args=(stream, None, None, circular_queue))
    # osx,
    p = multiprocessing.Process(
        target=webrtc_server,
        args=(
            None, stream.image_buffer, stream.info_buffer,
            None,
            stream_interaction.circular_queue.head_tail_buffer, 
            stream_interaction.circular_queue.buffers._buffers))
    p.start()
    stream_interaction.start(ms=ms_interaction)
    stream.init(ms_stream,)
    showm.start()
