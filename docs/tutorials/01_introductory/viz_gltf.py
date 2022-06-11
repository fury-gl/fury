"""
=======================
Visualizing a glTF file
=======================
In this tutorial, we will show how to display a glTF file in a scene.
"""

import os
from fury import utils, window, actor
from fury.gltf import glTFImporter
from urllib.request import urlretrieve

##############################################################################
# Create a scene.

scene = window.Scene()
scene.SetBackground(0.1, 0.1, 0.4)

##############################################################################
# Retrieving the gltf model and saving it.
url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Duck/glTF-Embedded/Duck.gltf"  # noqa

filename = url.split('/')
filename = os.path.basename(filename[len(filename)-1])

urlretrieve(url, filename=filename)

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

##############################################################################
# Removing the downloaded model
os.remove(filename)
