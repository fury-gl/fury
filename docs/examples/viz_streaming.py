from os.path import join as pjoin
from fury import actor, window, colormap as cmap
import numpy as np

###############################################################################
# Then let's download some available datasets.

from fury.data.fetcher import fetch_viz_wiki_nw

import multiprocessing
from fury.stream.servers.webrtc.server import webrtc_server

if __name__ == '__main__':


    ##############################################################################
    # First we will set the resolution which it'll be used by the streamer

    window_size = (200, 200)
    ##############################################################################
    # ms define the amount of mileseconds that will be used in the timer event.
    # Otherwise, if ms it's equal to zero the shared memory it's updated in each 
    # render event
    ms = 16
    ###############################################################################
    #
    arr = np.random.randint(
        0, 255, size=window_size[0]*window_size[1]*3).astype('uint8')
    #
    image_buffer = multiprocessing.RawArray('B', arr)
    #
    arr = None
    #
    arr = np.array([window_size[0], window_size[1], 3])
    info_buffer = multiprocessing.RawArray('I', arr)
    arr = None
    ###########################################################################
    #
    p = multiprocessing.Process(
        target=webrtc_server, args=(image_buffer, info_buffer))
    p.start()

    #
    files, folder = fetch_viz_wiki_nw()
    categories_file, edges_file, positions_file = sorted(files.keys())
    positions = np.loadtxt(pjoin(folder, positions_file))
    categories = np.loadtxt(pjoin(folder, categories_file), dtype=str)
    edges = np.loadtxt(pjoin(folder, edges_file), dtype=int)
    category2index = {category: i
                    for i, category in enumerate(np.unique(categories))}

    index2category = np.unique(categories)

    categoryColors = cmap.distinguishable_colormap(nb_colors=len(index2category))

    colors = np.array([categoryColors[category2index[category]]
                    for category in categories])
    radii = 1 + np.random.rand(len(positions))

    edgesPositions = []
    edgesColors = []
    for source, target in edges:
        edgesPositions.append(np.array([positions[source], positions[target]]))
        edgesColors.append(np.array([colors[source], colors[target]]))

    edgesPositions = np.array(edgesPositions)
    edgesColors = np.average(np.array(edgesColors), axis=1)

    sphere_actor = actor.sdf(
        centers=positions,
        colors=colors,
        scales=radii*0.5,)

    lines_actor = actor.line(edgesPositions,
                            colors=edgesColors,
                            opacity=0.1,
                            )
    scene = window.Scene()

    scene.add(lines_actor)
    scene.add(sphere_actor)

    interactive = False

    scene.set_camera(position=(0, 0, 1000), focal_point=(0.0, 0.0, 0.0),
                    view_up=(0.0, 0.0, 0.0))
                    
    showm = window.ShowManager(scene, reset_camera=False, size=(
        window_size[0], window_size[1]), order_transparent=False,
        # multi_samples=8
    )

    from fury.stream.client import FuryStreamClient
    showm.initialize()
    stream = FuryStreamClient(
        showm, False, image_buffer=image_buffer, info_buffer=info_buffer)
    stream.init(ms,)
    showm.start()
