"""
==========================================
Visualizing a glTF file with PBR materials
==========================================
In this tutorial, we will show how to display a glTF file that uses PBR
materials.
"""

from fury import window
from fury.data import fetch_gltf, read_viz_gltf
from fury.gltf import glTF

##############################################################################
# Create a scene.

scene = window.Scene()
scene.SetBackground(0.19, 0.21, 0.26)

##############################################################################
# Fetch and read the glTF file 'DamagedHelmet'.
fetch_gltf('DamagedHelmet')
filename = read_viz_gltf('DamagedHelmet')

##############################################################################
# Create a glTF object from the file and apply normals to the geometry.
gltf_obj = glTF(filename, apply_normals=True)

##############################################################################
# Extract actors representing the geometry from the glTF object.
actors = gltf_obj.actors()

##############################################################################
# Add all the actors representing the geometry to the scene.
scene.add(*actors)

##############################################################################
# Applying camera from the glTF object to the scene.

cameras = gltf_obj.cameras
if cameras:
    scene.SetActiveCamera(cameras[0])

interactive = 6

if interactive:
    window.show(scene, size=(1280, 720))

window.record(scene, out_path='viz_gltf_PBR.png', size=(1280, 720))
