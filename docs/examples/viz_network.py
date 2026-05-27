"""
====================================
Force-Directed Network Visualization
====================================

This example demonstrates how to use the `Network` actor to visualize
graphs using a GPU-accelerated Fruchterman-Reingold force-directed layout.

It supports two modes:
1. Dummy: Generates a random synthetic graph.
2. File: Loads a graph from .gexf, .gml, or .xnet files.
"""

import numpy as np
from fury import window, ui
from fury.network import Network

N_NODES = 500
N_EDGES = 1000
K_DISTANCE = 20.0
SIM_SPEED = 0.01

# To load from file, uncomment the lines below:
# from fury.io import load_network
# nodes_xyz, edges_indices, colors = load_network("path/to/network/file")
# network_actor = Network(
#     nodes=nodes_xyz, edges=edges_indices, colors=colors, k=K_DISTANCE, speed=SIM_SPEED
# )

nodes = (np.random.rand(N_NODES, 3) - 0.5) * 100
edges = np.random.randint(0, N_NODES, size=(N_EDGES, 2))
edges = edges[edges[:, 0] != edges[:, 1]]
colors = np.random.rand(N_NODES, 4).astype(np.float32)

network_actor = Network(
    nodes=nodes, edges=edges, colors=colors, k=K_DISTANCE, speed=SIM_SPEED
)

scene = window.Scene()
scene.background = (0.1, 0.1, 0.1)
scene.add(network_actor)

info_text = f"Nodes: {network_actor.n_nodes}\nEdges: {network_actor.n_edges}\n"
ui_label = ui.TextBlock2D(
    text=info_text, position=(10, 10), font_size=16, dynamic_bbox=True
)
scene.add(ui_label)

if __name__ == "__main__":
    show_manager = window.ShowManager(scene=scene, title="FURY Network")
    show_manager.start()
