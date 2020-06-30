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
# loaded texture. Add the actor to the scene.

earth_actor = actor.texture_on_sphere(image)
scene.add(earth_actor)

##############################################################################
# Do the same for the sun and the rest of the planets.

filename = read_viz_textures("8k_mercury.jpg")
image = io.load_image(filename)
mercury_actor = actor.texture_on_sphere(image)
scene.add(mercury_actor)

filename = read_viz_textures("8k_venus_surface.jpg")
image = io.load_image(filename)
venus_actor = actor.texture_on_sphere(image)
scene.add(venus_actor)

filename = read_viz_textures("8k_mars.jpg")
image = io.load_image(filename)
mars_actor = actor.texture_on_sphere(image)
scene.add(mars_actor)

filename = read_viz_textures("jupiter.jpg")
image = io.load_image(filename)
jupiter_actor = actor.texture_on_sphere(image)
scene.add(jupiter_actor)

utils.rotate(jupiter_actor, (90, 1, 0, 0))

filename = read_viz_textures("8k_saturn.jpg")
image = io.load_image(filename)
saturn_actor = actor.texture_on_sphere(image)
scene.add(saturn_actor)

# Add saturn's rings as a superquadratic

filename = read_viz_textures("2k_uranus.jpg")
image = io.load_image(filename)
uranus_actor = actor.texture_on_sphere(image)
scene.add(uranus_actor)

filename = read_viz_textures("2k_neptune.jpg")
image = io.load_image(filename)
neptune_actor = actor.texture_on_sphere(image)
scene.add(neptune_actor)

##############################################################################
# Lastly, create an actor for the sun.

filename = read_viz_textures("8k_sun.jpg")
image = io.load_image(filename)
sun_actor = actor.texture_on_sphere(image)
scene.add(sun_actor)

##############################################################################
# Next, change the positions and scales of the planets according to their
# position and size within the solar system.

sun_actor.SetScale(10, 10, 10)
mercury_actor.SetScale(0.4, 0.4, 0.4)
venus_actor.SetScale(0.6, 0.6, 0.6)
earth_actor.SetScale(0.4, 0.4, 0.4)
mars_actor.SetScale(0.8, 0.8, 0.8)
jupiter_actor.SetScale(2, 2, 2)
saturn_actor.SetScale(2, 2, 2)
uranus_actor.SetScale(1, 1, 1)
neptune_actor.SetScale(1, 1, 1)

mercury_actor.SetPosition(7, 0, 0)
venus_actor.SetPosition(9, 0, 0)
earth_actor.SetPosition(11, 0, 0)
mars_actor.SetPosition(13, 0, 0)
jupiter_actor.SetPosition(16, 0, 0)
saturn_actor.SetPosition(19, 0, 0)
uranus_actor.SetPosition(22, 0, 0)
neptune_actor.SetPosition(25, 0, 0)

##############################################################################
# Next, let's define the gravitational constants for each of these planets.
# This will allow us to visualize the orbit of each planet in our solar
# system. The gravitational constant, G, is measured in meters per second
# squared.

    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/

g_mercury = 3.7
g_venus = 8.9
g_earth = 9.8
g_mars = 3.7
g_jupiter = 23.1
g_saturn = 9.0
g_uranus = 8.7
g_neptune = 11.0

##############################################################################
# Also define the mass and orbital radii of each of the planets.

constant = np.power(10, 24)
m_mercury = 0.330 * constant
m_venus = 4.87 * constant
m_earth = 5.97 * constant
m_mars = 0.642 * constant
m_jupiter = 1898 * constant
m_saturn = 568 * constant
m_uranus = 86.8 * constant
m_neptune = 102 * constant

r_mercury = 7
r_venus = 9
r_earth = 11
r_mars = 13
r_jupiter = 16
r_saturn = 19
r_uranus = 22
r_neptune = 25

##############################################################################
# Let's define two functions that will help us calculate the position of each
# planets as it orbits around the sun: ``get_orbit_period`` and
# ``get_orbital_position``. The orbital period for each planet is a constant,
# so assign these values to a corresponding variable.

def get_orbit_period(radius, gravity):
    temp = np.sqrt(np.power(radius, 3)/gravity)
    return 2*np.pi * temp

orbit_period_mercury = get_orbit_period(r_mercury, g_mercury)
orbit_period_venus = get_orbit_period(r_venus, g_venus)
orbit_period_earth = get_orbit_period(r_earth, g_earth)
orbit_period_mars = get_orbit_period(r_mars, g_mars)
orbit_period_jupiter = get_orbit_period(r_jupiter, g_jupiter)
orbit_period_saturn = get_orbit_period(r_saturn, g_saturn)
orbit_period_uranus = get_orbit_period(r_uranus, g_uranus)
orbit_period_neptune = get_orbit_period(r_neptune, g_neptune)

def get_orbital_position(radius, time, gravity):
    orbit_period = get_orbit_period(radius, gravity)
    x = radius * np.cos((2*np.pi*time)/orbit_period)
    y = radius * np.sin((2*np.pi*time)/orbit_period)
    return (x, y)

##############################################################################
# Let's change the camera position to visualize the planets better.

scene.set_camera(position=(-20, 50, 100))

##############################################################################
# The ShowManager class is the interface between the scene, the window and the
# interactor.

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

##############################################################################
# Next, let's focus on creating the animation.
# We can determine the duration of animation with using the ``counter``.
# Use itertools to avoid global variables.

counter = itertools.count()

scene.add(actor.axes(scale=(12, 12, 12)))

##############################################################################
# Define the ``timer_callback`` function, which controls what events happen
# at certain times, using the counter. Redefine the position of each planet
# actor using ``get_orbital_position,`` assigning the x and y values of
# each planet's position with the newly calculated ones.

def timer_callback(_obj, _event):
    cnt = next(counter)
    showm.render()

    pos_mercury = get_orbital_position(r_mercury, cnt, g_mercury)
    mercury_actor.SetPosition(pos_mercury[0], 0, pos_mercury[1])

    pos_venus = get_orbital_position(r_venus, cnt, g_venus)
    venus_actor.SetPosition(pos_venus[0], 0, pos_venus[1])

    pos_earth = get_orbital_position(r_earth, cnt, g_earth)
    earth_actor.SetPosition(pos_earth[0], 0, pos_earth[1])

    pos_mars = get_orbital_position(r_mars, cnt, g_mars)
    mars_actor.SetPosition(pos_mars[0], 0, pos_mars[1])

    pos_jupiter = get_orbital_position(r_jupiter, cnt, g_jupiter)
    jupiter_actor.SetPosition(pos_jupiter[0], 0, pos_jupiter[1])

    pos_saturn = get_orbital_position(r_saturn, cnt, g_saturn)
    saturn_actor.SetPosition(pos_saturn[0], 0, pos_saturn[1])

    pos_uranus = get_orbital_position(r_uranus, cnt, g_uranus)
    uranus_actor.SetPosition(pos_uranus[0], 0, pos_uranus[1])

    pos_neptune = get_orbital_position(r_neptune, cnt, g_neptune)
    neptune_actor.SetPosition(pos_neptune[0], 0, pos_neptune[1])

    if cnt == 1000:
        showm.exit()

##############################################################################
# Watch your new animation take place!

showm.initialize()
showm.add_timer_callback(True, 35, timer_callback)
showm.start()
window.record(showm.scene, size=(900,768), out_path="viz_solar_system_animation.png")