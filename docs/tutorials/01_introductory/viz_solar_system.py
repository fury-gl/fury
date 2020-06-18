"""
===============
Texture Sphere Animation
===============
In this tutorial, we will create an animation of the solar system
using textured spheres.
"""

import numpy as np
from fury import window, actor, utils, primitive, io
import itertools
from fury.data.fetcher import read_viz_textures, fetch_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Next, load in a texture for each of the actors, starting with the Earth.
fetch_viz_textures()
filename = read_viz_textures("1_earth_8k.jpg")
image = io.load_image(filename)

##############################################################################
# Using ``actor.texture_on_sphere()``, create an earth_actor with your newly
# loaded texture.

earth_actor = actor.texture_on_sphere(image)

##############################################################################
# Do the same for the sun and the rest of the planets.

fetch_viz_textures()
filename = read_viz_textures("8k_mercury.jpg")
image = io.load_image(filename)
mercury_actor = actor.texture_on_sphere(image)

fetch_viz_textures()
filename = read_viz_textures("8k_venus_surface.jpg")
image = io.load_image(filename)
venus_actor = actor.texture_on_sphere(image)

fetch_viz_textures()
filename = read_viz_textures("8k_mars.jpg")
image = io.load_image(filename)
mars_actor = actor.texture_on_sphere(image)

fetch_viz_textures()
filename = read_viz_textures("8k_jupiter.jpg")
image = io.load_image(filename)
jupiter_actor = actor.texture_on_sphere(image)

fetch_viz_textures()
filename = read_viz_textures("8k_saturn.jpg")
image = io.load_image(filename)
saturn_actor = actor.texture_on_sphere(image)

# Add saturn's rings as a superquadratic

fetch_viz_textures()
filename = read_viz_textures("2k_uranus.jpg")
image = io.load_image(filename)
uranus_actor = actor.texture_on_sphere(image)

fetch_viz_textures()
filename = read_viz_textures("2k_neptune.jpg")
image = io.load_image(filename)
neptune_actor = actor.texture_on_sphere(image)

##############################################################################
# Lastly, create an actor for the sun.

fetch_viz_textures()
filename = read_viz_textures("8k_sun.jpg")
image = io.load_image(filename)
sun_actor = actor.texture_on_sphere(image)

##############################################################################
# Next, change the positions and scales of the planets according to their
# position and size within the solar system.

sun_actor.SetScale(5, 5, 5)
mercury_actor.SetScale(0.1, 0.1, 0.1)
venus_actor.SetScale(0.15, 0.15, 0.15)
earth_actor.SetScale(0.1, 0.1, 0.1)
jupiter_actor.SetScale(1, 1, 1)
saturn_actor.SetScale(1, 1, 1)
uranus_actor.SetScale(0.5, 0.5)
neptune_actor.SetScale(0.5, 0.5, 0.5)

mercury_actor.SetPosition(10, 0, 0)
venus_actor.SetPosition(20, 0, 0)
