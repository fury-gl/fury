import numpy as np
import vtk
import vtk.util.numpy_support as vtknp
from fury import actor, window, colormap as cmap

n = 10000

marker2id = {
            'o': 0, 's': 1, 'd': 2, '^': 3, 'p': 4,
            'h': 5, 's6': 6, 'x': 7, '+': 8}
markers = [
    np.random.choice(list(marker2id.keys()))
    for i in range(n)]

centers = np.random.normal(size=(n, 3), scale=10)

colors = np.random.uniform(size=(n, 3))
nodes_actor = actor.markers(
    centers,
    marker=markers,
    edge_width=.1,
    edge_color=[255, 255, 0],
    colors=colors,
    scales=.5,
)
nodes_3d_actor = actor.markers(
    centers+np.ones_like(centers)*25,
    marker='3d',
    colors=colors,
    scales=.5,
)

# this it's also possible 
# nodes_actor = actor.marker_billboard(centers, marker='o', )
scene = window.Scene()

scene.add(nodes_actor)
scene.add(nodes_3d_actor)

interactive = True

if interactive:
    window.show(scene, size=(600, 600))
