"""
===============
Sphere Texture
===============
In this tutorial, we will show how to create a sphere with a texture.
"""

import numpy as np
from fury import window, actor, utils, primitive, io
import itertools

##############################################################################
# Create a a scene to start.

scene = window.Scene()

##############################################################################
# Load an image (png, bmp, jpeg or jpg) using ``io.prim_sphere``.

filename = r"C:\Users\melin\Downloads\1_earth_8k.jpg"
image = io.load_image(filename)

##############################################################################
# Next, use ``actor.texture_on_sphere`` to add a sphere with the texture from
# your loaded image to the already existing scene.
# To add a texture to your scene as visualized on a plane, use
# ``actor.texture`` instead.

scene.add(actor.texture_on_sphere(image))

##############################################################################
# Lastly, record the scene. If you want to manipulate your new sphere with a
# texture, use ``window.show`` as commented out.

# window.show(scene, size=(600, 600), reset_camera=False)
window.record(scene, size=(900, 768), out_path="viz_texture.png")

