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
# Define information relevant for each planet actor including its
# texture name, relative position, and scale.

planets_data = [{'filename': '8k_mercury.jpg', 'position': 7,
                 'scale': (.4, .4, .4)},
                {'filename': '8k_venus_surface.jpg', 'position': 9,
                 'scale': (.6, .6, .6)},
                {'filename': '1_earth_8k.jpg', 'position': 11,
                 'scale': (.4, .4, .4)},
                {'filename': '8k_mars.jpg', 'position': 13,
                 'scale': (.8, .8, .8)},
                {'filename': 'jupiter.jpg', 'position': 16,
                 'scale': (2, 2, 2)},
                {'filename': '8k_saturn.jpg', 'position': 19,
                 'scale': (2, 2, 2)},
                {'filename': '8k_saturn_ring_alpha.png', 'position': 19,
                 'scale': (3, .5, 3)},
                {'filename': '2k_uranus.jpg', 'position': 22,
                 'scale': (1, 1, 1)},
                {'filename': '2k_neptune.jpg', 'position': 25,
                 'scale': (1, 1, 1)},
                {'filename': '8k_sun.jpg', 'position': 0,
                 'scale': (5, 5, 5)}]
fetch_viz_textures()

##############################################################################
# To take advantage of the previously defined data structure we are going to
# create an auxiliary function that will load and apply the respective
# texture, set its respective properties (relative position and scale),
# and add the actor to a previously created scene.


def init_planet(planet_data):
    """Initialize a planet actor.

    Parameters
    ----------
    planet_data : dict
        The planet_data is a dictionary, and the keys are filename(texture),
        position and scale.

    Returns
    -------
    planet_actor: actor
        The corresponding sphere actor with texture applied.
    """
    planet_file = read_viz_textures(planet_data['filename'])
    planet_image = io.load_image(planet_file)
    planet_actor = actor.texture_on_sphere(planet_image)
    planet_actor.SetPosition(planet_data['position'], 0, 0)
    planet_actor.SetScale(planet_data['scale'])
    scene.add(planet_actor)
    return planet_actor


##############################################################################
# Use the ``map`` function to create actors for each of the texture files
# in the ``planet_files`` list. Then, assign each actor to its corresponding
# actor in the list.

planet_actor_list = list(map(init_planet, planets_data))

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
# Define the gravitational constant G, the orbital radii of each of the
# planets, and the central mass of the sun. The gravity and mass will be
# used to calculate the orbital position, so multiply these two together to
# create a new constant, which we will call miu.

g_exponent = np.float_power(10, -11)
g_constant = 6.673*g_exponent

m_exponent = 1073741824   # np.power(10, 30)
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
    return x, y


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
    [planet_track.append([get_orbital_position(r_planet, i)[0],
                         0, get_orbital_position(r_planet, i)[1]])
     for i in range(cnt)]


##############################################################################
# First we are making a list that will contain radius from `planets_data`.
# `planet_tracks` is list of 8 empty lists for 8 different planets.
# And `planet_actors` will contain all the planet actors.

r_planets = list(set([planet_data['position']
                      for planet_data in planets_data]))
r_planets.remove(0)     # This will remove radius of orbit of sun

planet_tracks = [[], [], [], [], [], [], [], []]

planet_actors = [mercury_actor, venus_actor, earth_actor, mars_actor,
                 jupiter_actor, saturn_actor, uranus_actor, neptune_actor]

##############################################################################
# Here we are calculating and updating the path/orbit before animation starts.

for r_planet, planet_track in zip(r_planets, planet_tracks):
    calculate_path(r_planet, planet_track, r_planet * 85)

##############################################################################
# This is for orbit visualization. We are using line actor for orbits.
# After creating an actor we add it to the scene.

orbit_actor = actor.line(planet_tracks, colors=(1, 1, 1), linewidth=0.1)
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
        # if the planet is saturn then we also need to update the position
        # of its rings.
        if planet_actor == saturn_actor:
            pos_saturn = update_planet_position(19, saturn_actor, cnt)
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
