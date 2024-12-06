"""
===============
Sphere Texture
===============
In this tutorial, we will show how to create a sphere with a texture.
"""

import fury

##############################################################################
# Create a scene to start.

scene = fury.window.Scene()

##############################################################################
# Load an image (png, bmp, jpeg or jpg) using ``io.load_image``. In this
# example, we will use ``read_viz_textures`` to access an image of the
# Earth's surface from the fury Github after using ''fetch_viz_textures()''
# to download the available textures.

fury.data.fetch_viz_textures()
filename = fury.data.read_viz_textures("1_earth_8k.jpg")
image = fury.io.load_image(filename)

##############################################################################
# Next, use ``fury.actor.texture_on_sphere`` to add a sphere with the texture from
# your loaded image to the already existing scene.
# To add a texture to your scene as visualized on a plane, use
# ``fury.actor.texture`` instead.

scene.add(fury.actor.texture_on_sphere(image))

##############################################################################
# Lastly, record the scene, or set interactive to True if you would like to
# manipulate your new sphere.

interactive = False
if interactive:
    fury.window.show(scene, size=(600, 600), reset_camera=False)
fury.window.record(scene=scene, size=(900, 768), out_path="viz_texture.png")
