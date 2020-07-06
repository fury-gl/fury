"""
===============
Texture Sphere Animation
===============

In this tutorial, we will create an animation of the solar system
using textured spheres. We will also show how to manipulate the
position of these sphere actors in a timer_callback function
to simulate orbital motion.
"""

import numpy as np
import vtk
from fury import window, actor, utils, primitive, io
import itertools
from fury.data.fetcher import read_viz_textures, fetch_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Next, load in a texture for each of the actors, starting with the Earth.

fetch_viz_textures()
earth_filename = read_viz_textures("1_earth_8k.jpg")
earth_image = io.load_image(earth_filename)

##############################################################################
# Using ``actor.texture_on_sphere()``, create an earth_actor with your newly
# loaded texture. Add the actor to the scene.

earth_actor = actor.texture_on_sphere(earth_image)
scene.add(earth_actor)

##############################################################################
# Do the same for the rest of the planets.

mercury_filename = read_viz_textures("8k_mercury.jpg")
mercury_image = io.load_image(mercury_filename)
mercury_actor = actor.texture_on_sphere(mercury_image)
scene.add(mercury_actor)

venus_filename = read_viz_textures("8k_venus_surface.jpg")
venus_image = io.load_image(venus_filename)
venus_actor = actor.texture_on_sphere(venus_image)
scene.add(venus_actor)

earth_filename = read_viz_textures("8k_mars.jpg")
earth_image = io.load_image(earth_filename)
mars_actor = actor.texture_on_sphere(earth_image)
scene.add(mars_actor)

jupiter_filename = read_viz_textures("jupiter.jpg")
jupiter_image = io.load_image(jupiter_filename)
jupiter_actor = actor.texture_on_sphere(jupiter_image)
scene.add(jupiter_actor)

# Rotate this actor to correctly orient the texture
utils.rotate(jupiter_actor, (90, 1, 0, 0))

saturn_filename = read_viz_textures("8k_saturn.jpg")
saturn_image = io.load_image(saturn_filename)
saturn_actor = actor.texture_on_sphere(saturn_image)
scene.add(saturn_actor)

saturn_ring_filename = read_viz_textures("8k_saturn_ring_alpha.png")
saturn_ring_image = io.load_image(saturn_ring_filename)
saturn_rings_actor = actor.texture_on_sphere(saturn_ring_image)
scene.add(saturn_rings_actor)

uranus_filename = read_viz_textures("2k_uranus.jpg")
uranus_image = io.load_image(uranus_filename)
uranus_actor = actor.texture_on_sphere(uranus_image)
scene.add(uranus_actor)

neptune_filename = read_viz_textures("2k_neptune.jpg")
neptune_image = io.load_image(neptune_filename)
neptune_actor = actor.texture_on_sphere(neptune_image)
scene.add(neptune_actor)

##############################################################################
# Lastly, create an actor for the sun.

sun_filename = read_viz_textures("8k_sun.jpg")
sun_image = io.load_image(sun_filename)
sun_actor = actor.texture_on_sphere(sun_image)
scene.add(sun_actor)

##############################################################################
# Next, change the positions and scales of the planets according to their
# position and size within the solar system (relatively). For the purpose
# of this tutorial, planet sizes and positions will not be completely
# accurate.

sun_actor.SetScale(10, 10, 10)
mercury_actor.SetScale(0.4, 0.4, 0.4)
venus_actor.SetScale(0.6, 0.6, 0.6)
earth_actor.SetScale(0.4, 0.4, 0.4)
mars_actor.SetScale(0.8, 0.8, 0.8)
jupiter_actor.SetScale(2, 2, 2)
saturn_actor.SetScale(2, 2, 2)
saturn_rings_actor.SetScale(3, 0.5, 3)
uranus_actor.SetScale(1, 1, 1)
neptune_actor.SetScale(1, 1, 1)

mercury_actor.SetPosition(7, 0, 0)
venus_actor.SetPosition(9, 0, 0)
earth_actor.SetPosition(11, 0, 0)
mars_actor.SetPosition(13, 0, 0)
jupiter_actor.SetPosition(16, 0, 0)
saturn_actor.SetPosition(19, 0, 0)
saturn_rings_actor.SetPosition(19, 0, 0)
uranus_actor.SetPosition(22, 0, 0)
neptune_actor.SetPosition(25, 0, 0)

##############################################################################
# Next, let's define the gravitational constants for each of these planets.
# This will allow us to visualize the orbit of each planet in our solar
# system. The gravitational constant, G, is measured in meters per second
# squared. (https://nssdc.gsfc.nasa.gov/planetary/factsheet/)

g_mercury = 3.7
g_venus = 8.9
g_earth = 9.8
g_mars = 3.7
g_jupiter = 23.1
g_saturn = 9.0
g_uranus = 8.7
g_neptune = 11.0

##############################################################################
# Also define the orbital radii of each of the planets.

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
# planet as it orbits around the sun: ``get_orbit_period`` and
# ``get_orbital_position``. For the purpose of this tutorial, we will not be
# using the mass of each planet to calculate their orbital period, and
# instead will assume them all to be 1. This will improve the orbit
# visualization for the planets.


def get_orbit_period(radius, gravity):
    temp = np.sqrt(np.power(radius, 3)/gravity)
    return 2*np.pi * temp


def get_orbital_position(radius, time, gravity):
    orbit_period = get_orbit_period(radius, gravity)
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
# To track and visualize the orbital paths of the planets, we will create
# several new variables to map each planet's orbits using ``actor.dots``.
# It is important to define the track variable for each planet as global,
# allowing it to be used within the ``timer_callback`` function.

mercury_points_orbit = np.zeros((250, 3), dtype='f8')
global mercury_track
mercury_track = []
mercury_orbit_actor = actor.dots(mercury_points_orbit, color=(1, 1, 1),
                                 opacity=1, dot_size=2)
scene.add(mercury_orbit_actor)
positions_mercury = utils.vertices_from_actor(mercury_orbit_actor)

venus_points_orbit = np.zeros((250, 3), dtype='f8')
global venus_track
venus_track = []
venus_orbit_actor = actor.dots(venus_points_orbit, color=(1, 1, 1),
                               opacity=1, dot_size=2)
scene.add(venus_orbit_actor)
positions_venus = utils.vertices_from_actor(venus_orbit_actor)

earth_points_orbit = np.zeros((250, 3), dtype='f8')
global earth_track
earth_track = []
earth_orbit_actor = actor.dots(earth_points_orbit, color=(1, 1, 1),
                               opacity=1, dot_size=2)
scene.add(earth_orbit_actor)
positions_earth = utils.vertices_from_actor(earth_orbit_actor)

mars_points_orbit = np.zeros((250, 3), dtype='f8')
global mars_track
mars_track = []
mars_orbit_actor = actor.dots(mars_points_orbit, color=(1, 1, 1),
                              opacity=1, dot_size=2)
scene.add(mars_orbit_actor)
positions_mars = utils.vertices_from_actor(mars_orbit_actor)

jupiter_points_orbit = np.zeros((250, 3), dtype='f8')
global jupiter_track
jupiter_track = []
jupiter_orbit_actor = actor.dots(jupiter_points_orbit, color=(1, 1, 1),
                                 opacity=1, dot_size=2)
scene.add(jupiter_orbit_actor)
positions_jupiter = utils.vertices_from_actor(jupiter_orbit_actor)

saturn_points_orbit = np.zeros((250, 3), dtype='f8')
global saturn_track
saturn_track = []
saturn_orbit_actor = actor.dots(saturn_points_orbit, color=(1, 1, 1),
                                opacity=1, dot_size=2)
scene.add(saturn_orbit_actor)
positions_saturn = utils.vertices_from_actor(saturn_orbit_actor)

uranus_points_orbit = np.zeros((250, 3), dtype='f8')
global uranus_track
uranus_track = []
uranus_orbit_actor = actor.dots(uranus_points_orbit, color=(1, 1, 1),
                                opacity=1, dot_size=2)
scene.add(uranus_orbit_actor)
positions_uranus = utils.vertices_from_actor(uranus_orbit_actor)

neptune_points_orbit = np.zeros((250, 3), dtype='f8')
global neptune_track
neptune_track = []
neptune_orbit_actor = actor.dots(neptune_points_orbit, color=(1, 1, 1),
                                 opacity=1, dot_size=2)
scene.add(neptune_orbit_actor)
positions_neptune = utils.vertices_from_actor(neptune_orbit_actor)

##############################################################################
# Define the ``timer_callback`` function, which controls what events happen
# at certain times, using the counter. Redefine the position of each planet
# actor using ``get_orbital_position,`` assigning the x and y values of
# each planet's position with the newly calculated ones. Append each new
# planet position to its corresponding track array.


def timer_callback(_obj, _event):
    global mercury_track, venus_track, earth_track, mars_track, jupiter_track
    global saturn_track, uranus_track, neptune_track
    cnt = next(counter)
    showm.render()

    pos_mercury = get_orbital_position(r_mercury, cnt, g_mercury)
    mercury_actor.SetPosition(pos_mercury[0], 0, pos_mercury[1])
    mercury_track.append([pos_mercury[0], 0, pos_mercury[1]])

    pos_venus = get_orbital_position(r_venus, cnt, g_venus)
    venus_actor.SetPosition(pos_venus[0], 0, pos_venus[1])
    venus_track.append([pos_venus[0], 0, pos_venus[1]])

    pos_earth = get_orbital_position(r_earth, cnt, g_earth)
    earth_actor.SetPosition(pos_earth[0], 0, pos_earth[1])
    earth_track.append([pos_earth[0], 0, pos_earth[1]])

    pos_mars = get_orbital_position(r_mars, cnt, g_mars)
    mars_actor.SetPosition(pos_mars[0], 0, pos_mars[1])
    mars_track.append([pos_mars[0], 0, pos_mars[1]])

    pos_jupiter = get_orbital_position(r_jupiter, cnt, g_jupiter)
    jupiter_actor.SetPosition(pos_jupiter[0], 0, pos_jupiter[1])
    jupiter_track.append([pos_jupiter[0], 0, pos_jupiter[1]])

    pos_saturn = get_orbital_position(r_saturn, cnt, g_saturn)
    saturn_actor.SetPosition(pos_saturn[0], 0, pos_saturn[1])
    saturn_rings_actor.SetPosition(pos_saturn[0], 0, pos_saturn[1])
    saturn_track.append([pos_saturn[0], 0, pos_saturn[1]])

    pos_uranus = get_orbital_position(r_uranus, cnt, g_uranus)
    uranus_actor.SetPosition(pos_uranus[0], 0, pos_uranus[1])
    uranus_track.append([pos_uranus[0], 0, pos_uranus[1]])

    pos_neptune = get_orbital_position(r_neptune, cnt, g_neptune)
    neptune_actor.SetPosition(pos_neptune[0], 0, pos_neptune[1])
    neptune_track.append([pos_neptune[0], 0, pos_neptune[1]])

    if cnt == 249:
        positions_mercury[:] = np.array(mercury_track)
        utils.update_actor(mercury_orbit_actor)
        mercury_track = []

        positions_venus[:] = np.array(venus_track)
        utils.update_actor(venus_orbit_actor)
        venus_track = []

        positions_earth[:] = np.array(earth_track)
        utils.update_actor(earth_orbit_actor)
        earth_track = []

        positions_mars[:] = np.array(mars_track)
        utils.update_actor(mars_orbit_actor)
        mars_track = []

        positions_jupiter[:] = np.array(jupiter_track)
        utils.update_actor(jupiter_orbit_actor)
        jupiter_track = []

        positions_saturn[:] = np.array(saturn_track)
        utils.update_actor(saturn_orbit_actor)
        saturn_track = []

        positions_uranus[:] = np.array(uranus_track)
        utils.update_actor(uranus_orbit_actor)
        uranus_track = []

        positions_neptune[:] = np.array(neptune_track)
        utils.update_actor(neptune_orbit_actor)
        neptune_track = []

    if cnt == 1000:
        showm.exit()


##############################################################################
# Watch the planets orbit the sun in your new animation!

showm.initialize()
showm.add_timer_callback(True, 35, timer_callback)
showm.start()

window.record(showm.scene, size=(900, 768),
            out_path="viz_solar_system_animation.png")
