"""
====================================
Streaming FURY with WebRTC/MJPEG
====================================
"""

import multiprocessing
from os.path import join as pjoin

# if this example it's not working for you and you're using MacOs
# uncomment the following line
# multiprocessing.set_start_method('spawn')
import numpy as np

import fury

if __name__ == "__main__":
    interactive = False
    ###########################################################################
    # First we will set the resolution which it'll be used by the streamer

    window_size = (400, 400)

    files, folder = fury.data.fetch_viz_wiki_nw()
    categories_file, edges_file, positions_file = sorted(files.keys())
    positions = np.loadtxt(pjoin(folder, positions_file))
    categories = np.loadtxt(pjoin(folder, categories_file), dtype=str)
    edges = np.loadtxt(pjoin(folder, edges_file), dtype=int)
    category2index = {category: i for i, category in enumerate(np.unique(categories))}

    index2category = np.unique(categories)

    categoryColors = fury.colormap.distinguishable_colormap(
        nb_colors=len(index2category)
    )

    colors = np.array(
        [categoryColors[category2index[category]] for category in categories]
    )
    radii = 1 + np.random.rand(len(positions))

    edgesPositions = []
    edgesColors = []
    for source, target in edges:
        edgesPositions.append(np.array([positions[source], positions[target]]))
        edgesColors.append(np.array([colors[source], colors[target]]))

    edgesPositions = np.array(edgesPositions)
    edgesColors = np.average(np.array(edgesColors), axis=1)

    sphere_actor = fury.actor.sdf(
        centers=positions,
        colors=colors,
        primitives="sphere",
        scales=radii * 0.5,
    )

    lines_actor = fury.actor.line(
        edgesPositions,
        colors=edgesColors,
        opacity=0.1,
    )
    scene = fury.window.Scene()

    scene.add(lines_actor)
    scene.add(sphere_actor)

    scene.set_camera(
        position=(0, 0, 1000), focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 0.0, 0.0)
    )

    showm = fury.window.ShowManager(
        scene=scene,
        reset_camera=False,
        size=(window_size[0], window_size[1]),
        order_transparent=False,
    )

    ###########################################################################
    # ms define the amount of mileseconds that will be used in the timer event.

    ms = 0

    stream = fury.stream.FuryStreamClient(showm, use_raw_array=True)
    p = multiprocessing.Process(
        target=fury.stream.server.web_server_raw_array,
        args=(
            stream.img_manager.image_buffers,
            stream.img_manager.info_buffer,
        ),
    )
    p.start()

    stream.start(
        ms=ms,
    )
    if interactive:
        showm.start()
    stream.stop()
    stream.cleanup()

    fury.window.record(
        scene=showm.scene, size=window_size, out_path="viz_no_interaction.png"
    )
