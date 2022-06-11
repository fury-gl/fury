"""
=======================
Visualizing a glTF file 
=======================
In this tutorial, we will show how to display a glTF file in a scene.
"""

from fury import utils, window, actor
from fury.gltf import glTFImporter

##############################################################################
# Create a scene.

scene = window.Scene()
scene.SetBackground(0.1, 0.1, 0.4)

##############################################################################
# Define the filename as path to your local glTF file.

filename = "local-glTF/glTF-samples/suzanne/Suzanne.gltf"

##############################################################################
# Initialize the glTFImporter and get actors using `get_actors` method.
# Note: You can always manulally create actor from polydata, and apply texture
# or materials manually afterwards.

importer = glTFImporter(filename)
actors = importer.get_actors()

##############################################################################
# Add all the actor from list of actors to the scene.

for actor in actors:
    scene.add(actor)

interactive = True

if interactive:
    window.show(scene, size=(1280, 720))

window.record(scene, out_path='viz_gltf.png', size=(1280, 720))
