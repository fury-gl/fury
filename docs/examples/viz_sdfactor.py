"""
===================
Visualize SDF Actor
===================
Here is a simple tutorial that shows how to visualize SDF primitives using
FURY.

SDFs or Signed-distance functions when passed the coordinates of a point in
space, return the shortest distance between that point and some surface.
This property of SDFs can be used to model 3D geometry at a faster rate
compared to traditional polygons based modeling.

In this example we use the raymarching algorithm to render the SDF primitives
shapes using shaders
"""

import numpy as np

import fury

###############################################################################
# Lets define variables for the SDF Actor

dirs = np.random.rand(4, 3)
colors = np.random.rand(4, 3) * 255
centers = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0], [0, 1, 0]])
scales = np.random.rand(4, 1)


###############################################################################
# Create SDF Actor

sdfactor = fury.actor.sdf(
    centers=centers,
    directions=dirs,
    colors=colors,
    primitives=["sphere", "torus", "ellipsoid", "capsule"],
    scales=scales,
)

##############################################################################
# Create a scene

scene = fury.window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(sdfactor)


###############################################################################
# Show Manager
#
# Since all the elements have been initialised ,we add them to the show
# manager.

current_size = (1024, 720)
showm = fury.window.ShowManager(scene=scene,
                                size=current_size,
                                title="Visualize SDF Actor")

interactive = False

if interactive:
    showm.start()

fury.window.record(scene=scene, out_path="viz_sdfactor.png", size=current_size)
