"""
=======================
Solar System Animation
=======================

In this tutorial, we will create an animation of the solar system
using textured spheres. We will also show how to manipulate the
position of these sphere actors in a timer_callback function
to simulate orbital motion.
"""

import numpy as np
from fury import window, actor, utils, io
import itertools
from fury.data import read_viz_textures, fetch_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Load in a texture for each of the actors. To do this, we will create
# a function called ``init_planet``, which will initialize each planet actor
# given its corresponding filename and actor name. It will also add each
# actor to the scene that has already been created.

planet_filenames = ["8k_mercury.jpg", "8k_venus_surface.jpg",
                    "1_earth_8k.jpg", "8k_mars.jpg", "jupiter.jpg",
                    "8k_saturn.jpg", "8k_saturn_ring_alpha.png",
                    "2k_uranus.jpg", "2k_neptune.jpg",
                    "8k_sun.jpg"]
fetch_viz_textures()


def init_planet(filename):
    """Initialize a planet actor.

    Parameters
    ----------
    filename : str
        The filename for the corresponding planet texture.

    Returns
    -------
    planet_actor: actor
        The corresponding sphere actor with texture applied.
    """
    planet_file = read_viz_textures(filename)
    planet_image = io.load_image(planet_file)
    planet_actor = actor.texture_on_sphere(planet_image)
    scene.add(planet_actor)
    return planet_actor


##############################################################################
# Use the ``map`` function to create actors for each of the texture files
# in the ``planet_files`` list. Then, assign each actor to its corresponding
# actor in the list.

planet_actor_list = list(map(init_planet, planet_filenames))

mercury_actor = planet_actor_list[0]
venus_actor = planet_actor_list[1]
earth_actor = planet_actor_list[2]
mars_actor = planet_actor_list[3]
jupiter_actor = planet_actor_list[4]
saturn_actor = planet_actor_list[5]
saturn_rings_actor = planet_actor_list[6]
uranus_actor = planet_actor_list[7]
neptune_actor = planet_actor_list[8]
sun_actor = planet_actor_list[9]

# Rotate this actor to correctly orient the texture
utils.rotate(jupiter_actor, (90, 1, 0, 0))

##############################################################################
# Next, change the positions and scales of the planets according to their
# position and size within the solar system (relatively). For the purpose
# of this tutorial, planet sizes and positions will not be completely
# accurate.

r_mercury = 7
r_venus = 9
r_earth = 11
r_mars = 13
r_jupiter = 16
r_saturn = 19
r_uranus = 22
r_neptune = 25

sun_actor.SetScale(5, 5, 5)
mercury_actor.SetScale(0.4, 0.4, 0.4)
venus_actor.SetScale(0.6, 0.6, 0.6)
earth_actor.SetScale(0.4, 0.4, 0.4)
mars_actor.SetScale(0.8, 0.8, 0.8)
jupiter_actor.SetScale(2, 2, 2)
saturn_actor.SetScale(2, 2, 2)
saturn_rings_actor.SetScale(3, 0.5, 3)
uranus_actor.SetScale(1, 1, 1)
neptune_actor.SetScale(1, 1, 1)

mercury_actor.SetPosition(r_mercury, 0, 0)
venus_actor.SetPosition(r_venus, 0, 0)
earth_actor.SetPosition(r_earth, 0, 0)
mars_actor.SetPosition(r_mars, 0, 0)
jupiter_actor.SetPosition(r_jupiter, 0, 0)
saturn_actor.SetPosition(r_saturn, 0, 0)
saturn_rings_actor.SetPosition(r_saturn, 0, 0)
uranus_actor.SetPosition(r_uranus, 0, 0)
neptune_actor.SetPosition(r_neptune, 0, 0)

##############################################################################
# Define the gravitational constant G, the orbital radii of each of the
# planets, and the central mass of the sun. The gravity and mass will be used
# to calculate the orbital position, so multiply these two together to create
# a new constant, which we will call miu.

g_exponent = np.float_power(10, -11)
g_constant = 6.673*g_exponent

m_exponent = 1073741824                                    # np.power(10, 30)
m_constant = 1.989*m_exponent

miu = m_constant*g_constant

##############################################################################
# Let's define two functions that will help us calculate the position of each
# planet as it orbits around the sun: ``get_orbit_period`` and
# ``get_orbital_position``, using the constant miu and the orbital radii
# of each planet.


def get_orbit_period(radius):
    return 2 * np.pi * np.sqrt(np.power(radius, 3)/miu)


def get_orbital_position(radius, time):
    orbit_period = get_orbit_period(radius)
    x = radius * np.cos((2*np.pi*time)/orbit_period)
    y = radius * np.sin((2*np.pi*time)/orbit_period)
    return (x, y)


##############################################################################
# Let's change the camera position to visualize the planets better.

scene.set_camera(position=(-20, 60, 100))

##############################################################################
# Next, create a ShowManager object. The ShowManager class is the interface
# between the scene, the window and the interactor.

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

##############################################################################
# Next, let's focus on creating the animation.
# We can determine the duration of animation with using the ``counter``.
# Use itertools to avoid global variables.

counter = itertools.count()

##############################################################################
# Define one new function to use in ``timer_callback`` to update the planet
# positions ``update_planet_position``.


def update_planet_position(r_planet, planet_actor, cnt):
    pos_planet = get_orbital_position(r_planet, cnt)
    planet_actor.SetPosition(pos_planet[0], 0, pos_planet[1])
    return pos_planet


##############################################################################
# ``calculate_path`` function is for calculating the path/orbit
# of every planet.


def calculate_path(r_planet, planet_track, cnt):
    for i in range(cnt):
        pos_planet = get_orbital_position(r_planet, i)
        planet_track.append([pos_planet[0], 0, pos_planet[1]])


##############################################################################
# First we are making two lists that will contain radii and track's lists.
# `planet_tracks` is list of 8 empty list for 8 different planets.
# and `planet_actors` will contain all the planet actor.
# and then we are calculating and updating the path/orbit
# before animation starts.

r_planets = [r_mercury, r_venus, r_earth, r_mars,
                r_jupiter, r_saturn, r_uranus, r_neptune]
planet_tracks = [[], [], [], [], [], [], [], []]
planet_actors = [mercury_actor, venus_actor, earth_actor, mars_actor,
                jupiter_actor, saturn_actor, uranus_actor, neptune_actor]

for r_planet, planet_track in zip(r_planets, planet_tracks):
    calculate_path(r_planet, planet_track, r_planet * 85)

##############################################################################
# This is for orbit visualization. We are using line actor for orbits.
# After creating an actor we add it to the scene.

for planet_track in planet_tracks:
    orbit_actor = actor.line([planet_track], colors=(1, 1, 1), linewidth=0.1)
    scene.add(orbit_actor)

##############################################################################
# Define the ``timer_callback`` function, which controls what events happen
# at certain times, using the counter. Update the position of each planet
# actor using ``update_planet_position,`` assigning the x and y values of
# each planet's position with the newly calculated ones.


def timer_callback(_obj, _event):
    cnt = next(counter)
    showm.render()

    for r_planet, planet_actor in zip(r_planets, planet_actors):
        if r_planet == r_saturn:
            pos_saturn = update_planet_position(r_saturn, saturn_actor, cnt)
            saturn_rings_actor.SetPosition(pos_saturn[0], 0, pos_saturn[1])
        else:
            update_planet_position(r_planet, planet_actor, cnt)

    if cnt == 2000:
        showm.exit()


##############################################################################
# Watch the planets orbit the sun in your new animation!

showm.initialize()
showm.add_timer_callback(True, 5, timer_callback)
showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_solar_system_animation.png")
