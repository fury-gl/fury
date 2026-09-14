"""
=======================
Visualizing a glTF file
=======================

In this tutorial, we will show how to display a glTF file in a scene.

glTF 2.0 is the runtime asset format for 3D scenes: a single ``.gltf`` or
``.glb`` file carries the meshes, their physically based materials and
textures, the node hierarchy positioning them, and any cameras, lights and
animations the author defined.

``fury.gltf.glTF`` reads such a file and exposes what it found. Reading needs
the optional ``gltflib`` dependency, installed with ``pip install fury[gltf]``.
"""

#########################################################################################
# Import the required libraries.
import fury

#########################################################################################
# Retrieve the glTF model. ``fetch_gltf`` downloads a sample from the Khronos
# glTF-Sample-Models repository and caches it, and ``read_viz_gltf`` returns the
# path of the cached file. Call ``fury.data.list_gltf_sample_models()`` to see
# every sample available.
fury.data.fetch_gltf(name="Duck", mode="glTF")
filename = fury.data.read_viz_gltf("Duck")

#########################################################################################
# Initialize the glTF object and get the actors with the ``actors`` method.
#
# The actors are the nodes of the imported scene itself, with their node
# transforms already applied, so they arrive positioned relative to one another
# as the author placed them. They are FURY actors, so ``rotate``, ``translate``
# and ``scale`` work on them as they do on any actor you build yourself.
gltf_obj = fury.gltf.glTF(filename)
actors = gltf_obj.actors()

print(f"{filename} contains {len(actors)} actor(s)")

#########################################################################################
# Add the actors to the scene.
scene = fury.window.Scene(background=(0.05, 0.05, 0.08))
scene.add(*actors)

#########################################################################################
# Applying a camera from the file.
#
# glTF files may ship their own cameras, already positioned by the node they
# hang off. The Duck declares one, so we frame the scene with it rather than
# placing a camera by hand.
show_m = fury.window.ShowManager(scene=scene, size=(1280, 720), title="FURY glTF")

camera = show_m.screens[0].camera
camera.local.position = gltf_obj.cameras[0].world.position
camera.look_at((0, 0, 0))

#########################################################################################
# Start the rendering loop.
show_m.start()
