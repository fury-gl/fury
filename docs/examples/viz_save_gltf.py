"""
========================
Saving a Surface as glTF
========================

glTF 2.0 is a widely supported format for 3D models, stored either as a ``.gltf``
JSON file with a ``.bin`` buffer or as a single-file binary ``.glb``. This tutorial
builds a small colored surface, saves it with ``fury.io.save_gltf``, reads the written
file back with ``fury.io.read_gltf``, and renders the original and the saved copy side
by side.

Only geometry is written: triangles with optional per-vertex colors and normals.
Materials, textures, and animations are not saved.
"""

#########################################################################################
# Import the required libraries.
import os

import numpy as np

from fury import actor, window
from fury.io import read_gltf, save_gltf

#########################################################################################
# Build a surface as NumPy arrays: a square pyramid made of four base corners and an
# apex. Six triangles (four sides and two for the base) connect them, and every vertex
# gets one RGB color.
vertices = np.array(
    [[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5], [0, 1, 0]],
    dtype=np.float32,
)
faces = np.array(
    [[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0], [0, 1, 2], [0, 2, 3]],
    dtype=np.int32,
)
colors = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32
)

#########################################################################################
# Create the actor for the original surface and move it to the left.
original_actor = actor.surface(vertices, faces, colors=colors)
original_actor.local.position = (-1.0, 0.0, 0.0)

#########################################################################################
# Save the surface to the current working directory, so the file can be opened in any
# glTF viewer afterwards. A ``.glb`` path produces one binary file, while a ``.gltf``
# path would write JSON plus a ``.bin`` file next to it. ``normals`` can be passed too;
# here they are left out and the renderer derives the shading. The file is then read
# back to check what was stored.
gltf_path = os.path.abspath("pyramid.glb")
save_gltf(vertices, faces, gltf_path, colors=colors)
saved_vertices, saved_faces, saved_colors, saved_normals = read_gltf(gltf_path)

print(f"Saved {gltf_path}")
print(
    f"{len(saved_vertices)} vertices, {len(saved_faces)} faces, "
    f"colors: {saved_colors is not None}, normals: {saved_normals is not None}"
)

#########################################################################################
# Create the actor for the saved copy and move it to the right. Both pyramids should
# look identical.
saved_actor = actor.surface(saved_vertices, saved_faces, colors=saved_colors)
saved_actor.local.position = (1.0, 0.0, 0.0)

#########################################################################################
# Render both actors. The camera is framed automatically around the scene.
scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(original_actor, saved_actor)

show_m = window.ShowManager(scene=scene, size=(1000, 600), title="FURY glTF Tutorial")
show_m.start()
