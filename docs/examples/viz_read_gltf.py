"""
===================
Reading glTF Models
===================

glTF 2.0 stores a scene of meshes placed by node transforms, either as a ``.gltf``
JSON file with a ``.bin`` buffer or as a single-file binary ``.glb``. This tutorial
fetches a sample model, loads it with ``fury.io.read_gltf`` and renders it with
``actor.surface``.

``read_gltf`` applies the node transforms of the default scene and merges all of its
meshes into one set of ``(vertices, faces, colors, normals)`` arrays. ``colors`` and
``normals`` are ``None`` when the file does not provide them. Materials, textures, and
animations are not read.
"""

#########################################################################################
# Import the required libraries.
import numpy as np
import polyxios as px

from fury import actor, window
from fury.io import read_gltf

#########################################################################################
# Fetch the sample model via polyxios and load it. ``BoxVertexColors.glb`` is a cube
# with per-vertex colors and normals.
box_path = px.fetch("BoxVertexColors.glb")
box_vertices, box_faces, box_colors, box_normals = read_gltf(box_path)

print(f"BoxVertexColors.glb: {len(box_vertices)} vertices, {len(box_faces)} faces")

#########################################################################################
# Center the model and scale it to unit size.
box_vertices = box_vertices - box_vertices.mean(axis=0)
box_vertices = (box_vertices / np.max(np.abs(box_vertices))).astype(np.float32)

#########################################################################################
# Build the actor from the loaded arrays.
box_actor = actor.surface(
    box_vertices, box_faces, colors=box_colors, normals=box_normals
)

#########################################################################################
# Render the actor. The camera is framed automatically around the scene.
scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(box_actor)

show_m = window.ShowManager(scene=scene, size=(1000, 600), title="FURY glTF Reader")
show_m.start()
