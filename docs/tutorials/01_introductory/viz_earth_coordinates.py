
"""
============================
Earth Coordinate Conversion
============================

In this tutorial, we will show how to place actors on specific locations
on the surface of the Earth using a new function.
"""

from fury import window, actor, utils, io
from fury.data import read_viz_textures, fetch_viz_textures
import math
import numpy as np
import itertools

###############################################################################
# Create a new scene, and load in the image of the Earth using
# ``fetch_viz_textures`` and ``read_viz_textures``. We will use a 16k
# resolution texture for maximum detail.

scene = window.Scene()

fetch_viz_textures()
earth_file = read_viz_textures("1_earth_16k.jpg")
earth_image = io.load_image(earth_file)
earth_actor = actor.texture_on_sphere(earth_image)
scene.add(earth_actor)

###############################################################################
# Rotate the Earth to make sure the texture is correctly oriented. Change it's
# scale using ``actor.SetScale()``.

utils.rotate(earth_actor, (-90, 1, 0, 0))
utils.rotate(earth_actor, (180, 0, 1, 0))
earth_actor.SetScale(2, 2, 2)

###############################################################################
# Define the function to convert geographical coordinates of a location in
# latitude and longitude degrees to coordinates on the ``earth_actor`` surface.
# In this function, convert to radians, then to spherical coordinates, and
# lastly, to cartesian coordinates.


def latlong_coordinates(lat, lon):
    # Convert latitude and longitude to spherical coordinates
    degrees_to_radians = math.pi/180.0
    # phi = 90 - latitude
    phi = (90-lat)*degrees_to_radians
    # theta = longitude
    theta = lon*degrees_to_radians*-1
    # now convert to cartesian
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    # flipping z to y for FURY coordinates
    return (x, z, y)


###############################################################################
# Use this new function to place some sphere actors on several big cities
# around the Earth.

locationone = latlong_coordinates(40.730610, -73.935242)  # new york city, us
locationtwo = latlong_coordinates(39.916668, 116.383331)  # beijing, china
locationthree = latlong_coordinates(48.864716, 2.349014)  # paris, france

###############################################################################
# Set the centers, radii, and colors of these spheres, and create a new
# ``sphere_actor`` for each location to add to the scene.

centers = np.array([[*locationone], [*locationtwo], [*locationthree]])
colors = np.random.rand(3, 3)
radii = np.array([0.005, 0.005, 0.005])
sphere_actor = actor.sphere(centers, colors, radii)
scene.add(sphere_actor)

###############################################################################
# Create some text actors to add to the scene indicating each location and its
# geographical coordinates.

nyc_actor = actor.text_3d("New York City, New York\n40.7128° N, 74.0060° W",
                          (locationone[0]-0.04, locationone[1],
                           locationone[2]+0.07),
                          window.colors.white, 0.01)
paris_actor = actor.text_3d("Paris, France\n48.8566° N, 2.3522° E",
                            (locationthree[0]-0.04, locationthree[1],
                             locationthree[2]-0.07),
                            window.colors.white, 0.01)
beijing_actor = actor.text_3d("Beijing, China\n39.9042° N, 116.4074° E",
                              (locationtwo[0]-0.06, locationtwo[1],
                               locationtwo[2]-0.07),
                              window.colors.white, 0.01)
utils.rotate(paris_actor, (85, 0, 1, 0))
utils.rotate(beijing_actor, (180, 0, 1, 0))
utils.rotate(nyc_actor, (5, 1, 0, 0))

##############################################################################
# Create a ShowManager object, which acts as the interface between the scene,
# the window and the interactor.

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

###############################################################################
# Let's create a ``timer_callback`` function to add some animation to the
# Earth. Change the camera position and angle to fly over and zoom in on each
# new location.

counter = itertools.count()


def timer_callback(_obj, _event):
    cnt = next(counter)
    showm.render()
    if cnt == 0:
        scene.set_camera(position=(1.5, 3.5, 7.0))
    if cnt < 200 and cnt > 25:
        scene.zoom(1.015)
        scene.pitch(0.01)
    if cnt == 200:
        scene.add(nyc_actor)
    if cnt > 250 and cnt < 350:
        scene.zoom(0.985)
    if cnt > 350 and cnt < 425:
        scene.azimuth(1)
    if cnt > 425 and cnt < 525:
        scene.zoom(1.015)
        scene.pitch(0.011)
    if cnt == 525:
        scene.add(paris_actor)
    if cnt > 575 and cnt < 700:
        scene.zoom(0.985)
    if cnt > 700 and cnt < 820:
        scene.azimuth(1)
    if cnt > 820 and cnt < 930:
        scene.zoom(1.015)
    if cnt == 930:
        scene.add(beijing_actor)
    if cnt == 1000:
        showm.exit()


###############################################################################
# Initialize the ShowManager object, add the timer_callback, and watch the
# new animation take place!


showm.add_timer_callback(True, 25, timer_callback)
showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_earth_coordinates.png")
