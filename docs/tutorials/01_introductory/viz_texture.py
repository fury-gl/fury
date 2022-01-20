"""
===============
Sphere Texture
===============
In this tutorial, we will show how to create a sphere with a texture.
"""

from fury import window, actor, io
from fury.data import read_viz_textures, fetch_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Load an image (png, bmp, jpeg or jpg) using ``io.load_image``. In this
# example, we will use ``read_viz_textures`` to access an image of the
# Earth's surface from the fury Github after using ''fetch_viz_textures()''
# to download the available textures.

fetch_viz_textures()
filename = read_viz_textures("1_earth_8k.jpg")
image = io.load_image(filename)

##############################################################################
# Next, use ``actor.texture_on_sphere`` to add a sphere with the texture from
# your loaded image to the already existing scene.
# To add a texture to your scene as visualized on a plane, use
# ``actor.texture`` instead.

scene.add(actor.texture_on_sphere(image))

##############################################################################
# Lastly, record the scene, or set interactive to True if you would like to
# manipulate your new sphere.

interactive = False
if interactive:
    window.show(scene, size=(600, 600), reset_camera=False)
window.record(scene, size=(900, 768), out_path="viz_texture.png")
