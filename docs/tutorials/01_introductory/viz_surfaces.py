"""
==================
Visualize surfaces
==================

Here is a simple tutorial that shows how to visualize surfaces using DIPY. It
also shows how to load/save, get/set and update ``PolyData`` and show
surfaces.

``PolyData`` is a structure used by VTK to represent surfaces and other data
structures. Here we show how to visualize a simple cube but the same idea
should apply for any surface.
"""

import numpy as np

###############################################################################
# Import useful functions

from fury import window, utils
from fury.io import save_polydata, load_polydata
from fury.lib import PolyData

###############################################################################
# Create an empty ``PolyData``

my_polydata = PolyData()

###############################################################################
# Create a cube with vertices and triangles as numpy arrays

my_vertices = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 0.0, 1.0],
                       [1.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0]])
# the data type is needed to mention here, numpy.int64
my_triangles = np.array([[0, 6, 4],
                         [0, 2, 6],
                         [0, 3, 2],
                         [0, 1, 3],
                         [2, 7, 6],
                         [2, 3, 7],
                         [4, 6, 7],
                         [4, 7, 5],
                         [0, 4, 5],
                         [0, 5, 1],
                         [1, 5, 7],
                         [1, 7, 3]], dtype='i8')


###############################################################################
# Set vertices and triangles in the ``PolyData``

utils.set_polydata_vertices(my_polydata, my_vertices)
utils.set_polydata_triangles(my_polydata, my_triangles)

###############################################################################
# Save the ``PolyData``

file_name = "my_cube.vtk"
save_polydata(my_polydata, file_name)
print("Surface saved in " + file_name)

###############################################################################
# Load the ``PolyData``

cube_polydata = load_polydata(file_name)

###############################################################################
# add color based on vertices position

cube_vertices = utils.get_polydata_vertices(cube_polydata)
colors = cube_vertices * 255
utils.set_polydata_colors(cube_polydata, colors)

print("new surface colors")
print(utils.get_polydata_colors(cube_polydata))

###############################################################################
# Visualize surfaces

# get Actor
cube_actor = utils.get_actor_from_polydata(cube_polydata)

# Create a scene
scene = window.Scene()
scene.add(cube_actor)
scene.set_camera(position=(10, 5, 7), focal_point=(0.5, 0.5, 0.5))
scene.zoom(3)

# display
# window.show(scene, size=(600, 600), reset_camera=False)
window.record(scene, out_path='cube.png', size=(600, 600))
