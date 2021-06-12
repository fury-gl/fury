from os.path import join as pjoin
from fury import actor, window, colormap as cmap
import numpy as np

###############################################################################
# An example about how to use the streaming with the interaction running 
# in another process


import multiprocessing
from fury.stream.servers.webrtc.server import webrtc_server, interaction_server
from fury.stream.client import FuryStreamClient, FuryStreamInteraction
if __name__ == '__main__':

    window_size = (400, 400)
    max_window_size = (700, 700)
    # 0 ms_stream means that the frame will be sent to the server
    # right after the rendering
    ms_interaction = 1
    ms_stream = 16
    # max number of interactions to be stored inside the queue
    max_queue_size = 1000
    ##############################################################################

    centers = 1*np.array([
        [0, 0, 0],
        [-1, 0, 0],
        [1, 0, 0]
    ])
    centers2 = centers - np.array([[0, -1, 0]])
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

    stream_interaction = FuryStreamInteraction(
        showm, max_queue_size=max_queue_size)
    showm.initialize()

    stream = FuryStreamClient(
        showm, window_size, max_window_size=max_window_size)
    # linux
    # p = multiprocessing.Process(
    #     target=webrtc_server,
    #     args=(stream, None, None, circular_queue))
    # osx,
    p = multiprocessing.Process(
        target=webrtc_server,
        args=(
            None, stream.image_buffers,
            stream.image_buffer_names,
            stream.info_buffer,
            stream.info_buffer_name,
            None,)
    )
    p.start()
    p2 = multiprocessing.Process(
        target=interaction_server,
        args=(
            None,
            stream_interaction.circular_queue.head_tail_buffer,
            stream_interaction.circular_queue.buffers._buffers,
            8080, 'localhost')
    )
    p2.start()
    stream_interaction.start(ms=ms_interaction)
    stream.init(ms_stream,)
    showm.start()
    p.kill()
    p2.kill()
    stream.cleanup()

    # open a browser using the following the url
    # http://localhost:8000?interaction_addr=localhost:8080