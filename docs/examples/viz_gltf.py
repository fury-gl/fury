"""
=======================
Visualizing a glTF file
=======================
In this tutorial, we will show how to display a glTF file in a scene.
"""

import fury

##############################################################################
# Create a scene.

scene = fury.window.Scene()
scene.SetBackground(0.1, 0.1, 0.4)

##############################################################################
# Retrieving the gltf model.
fury.data.fetch_gltf("Duck", "glTF")
filename = fury.data.read_viz_gltf("Duck")

##############################################################################
# Initialize the glTF object and get actors using `actors` method.
# Note: You can always manually create actor from polydata, and apply texture
# or materials manually afterwards.
# Experimental: For smooth mesh/actor you can set `apply_normals=True`.

gltf_obj = fury.gltf.glTF(filename, apply_normals=False)
actors = gltf_obj.actors()

##############################################################################
# Add all the actor from list of actors to the scene.

scene.add(*actors)

##############################################################################
# Applying camera

cameras = gltf_obj.cameras
if cameras:
    scene.SetActiveCamera(cameras[0])

interactive = False

if interactive:
    fury.window.show(scene, size=(1280, 720))

fury.window.record(scene, out_path="viz_gltf.png", size=(1280, 720))
