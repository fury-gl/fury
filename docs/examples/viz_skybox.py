"""
===============
Adding a Skybox
===============

This example demonstrates how to use the add a skybox to a scene in FURY.

"""

import numpy as np

from fury.window import ShowManager, Scene
from fury.actor import sphere
from fury.data import read_viz_cubemap, fetch_viz_cubemaps
from fury.io import load_cube_map_texture


###############################################################################
# Let's fetch a skybox texture from the FURY data repository.

fetch_viz_cubemaps()

###############################################################################
# The following function returns the full path of the 6 images composing the
# skybox.

texture_files = read_viz_cubemap("skybox")

###############################################################################
# Now that we have the location of the textures, let's load them and create a
# Cube Map Texture object.

cube_map = load_cube_map_texture(texture_files)

###############################################################################
# The Scene object in FURY can handle cube map textures and extract light
# information from them, so it can be used to create more plausible materials
# interactions. The ``skybox`` parameter takes as input a cube map texture and
# performs the previously described process.

scene = Scene(skybox=cube_map)

sphere_actor = sphere(
    np.zeros((1, 3)),
    colors=(1, 0, 1, 1),
    radii=15.0,
    phi=48,
    theta=48,
)
scene.add(sphere_actor)


if __name__ == "__main__":
    show_m = ShowManager(scene=scene, title="FURY 2.0: Skybox Example")
    show_m.start()
