"""
============================================================
This examples shows how to visualize a large amount of labels
=============================================================

The goal of this example is to show how to visualize a large amount of labels
using the interdisciplinary map of journal networks.

"""

###############################################################################
# First, let's import some useful functions

from os.path import join as pjoin
from fury import actor, window, colormap as cmap
import numpy as np

###############################################################################
# Then let's download some available datasets.

from fury.data.fetcher import fetch_viz_wiki_nw

files, folder = fetch_viz_wiki_nw()
categories_file, edges_file, positions_file = sorted(files.keys())

###############################################################################
# We read our datasets

positions = np.loadtxt(pjoin(folder, positions_file))
categories = np.loadtxt(pjoin(folder, categories_file), dtype=str)
edges = np.loadtxt(pjoin(folder, edges_file), dtype=int)

###############################################################################
# We attribute a color to each category of our dataset which correspond to our
# nodes colors.

category2index = {category: i
                  for i, category in enumerate(np.unique(categories))}

index2category = np.unique(categories)

category_colors = cmap.distinguishable_colormap(nb_colors=len(index2category))

colors = np.array([category_colors[category2index[category]]
                   for category in categories])

###############################################################################
# We define our node size

radii = 1 + np.random.rand(len(positions))

###############################################################################
# Lets create our edges now. They will indicate a citation between two nodes.
# OF course, the colors of each edges will be an interpolation between the two
# node that it connects.

edges_positions = []
edges_colors = []
for source, target in edges:
    edges_positions.append(np.array([positions[source], positions[target]]))
    edges_colors.append(np.array([colors[source], colors[target]]))

edges_positions = np.array(edges_positions)
edges_colors = np.average(np.array(edges_colors), axis=1)

###############################################################################
# Our data preparation is ready, it is time to visualize them all. We start to
# build 2 actors that we represent our data : sphere_actor for the nodes and
# lines_actor for the edges.

sphere_actor = actor.markers(
    centers=positions,
    colors=colors,
    scales=radii*0.1,
)

lines_actor = actor.line(
    edges_positions,
    colors=edges_colors,
    opacity=0.1,
)

###############################################################################
# Now, we will create the  list of labels that will be displayed on the nodes.
#

labels = [
    f'{category} journal {i}' for i, category in enumerate(categories)
]
#############################################################################
# Finally, we create our network label actor.

my_text_actor = actor.bitmap_labels(
    positions, labels,
    y_offset_ratio=2,
    align='center', scales=.1)

###############################################################################
# All actors need to be added in a scene, so we build one and add our
# lines_actor and sphere_actor.

scene = window.Scene()

scene.add(lines_actor)
scene.add(sphere_actor)
scene.add(my_text_actor)

###############################################################################
# The final step ! Visualize and save the result of our creation! Please,
# switch interactive variable to True if you want to visualize it.
interactive = False

if interactive:
    window.show(scene, size=(600, 600))

window.record(scene, out_path='viz_huge_amount_of_labels.png', size=(600, 600))