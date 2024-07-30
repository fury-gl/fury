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

import fury

###############################################################################
# Import useful functions


###############################################################################
# Create an empty ``PolyData``

my_polydata = fury.lib.PolyData()

###############################################################################
# Create a cube with vertices and triangles as numpy arrays

my_vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)
# the data type is needed to mention here, numpy.int64
my_triangles = np.array(
    [
        [0, 6, 4],
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
        [1, 7, 3],
    ],
    dtype="i8",
)


###############################################################################
# Set vertices and triangles in the ``PolyData``

fury.utils.set_polydata_vertices(my_polydata, my_vertices)
fury.utils.set_polydata_triangles(my_polydata, my_triangles)

###############################################################################
# Save the ``PolyData``

file_name = "my_cube.vtk"
fury.io.save_polydata(my_polydata, file_name)
print("Surface saved in " + file_name)

###############################################################################
# Load the ``PolyData``

cube_polydata = fury.io.load_polydata(file_name)

###############################################################################
# add color based on vertices position

cube_vertices = fury.utils.get_polydata_vertices(cube_polydata)
colors = cube_vertices * 255
fury.utils.set_polydata_colors(cube_polydata, colors)

print("new surface colors")
print(fury.utils.get_polydata_colors(cube_polydata))

###############################################################################
# Visualize surfaces

# get Actor
cube_actor = fury.utils.get_actor_from_polydata(cube_polydata)

# Create a scene
scene = fury.window.Scene()
scene.add(cube_actor)
scene.set_camera(position=(10, 5, 7), focal_point=(0.5, 0.5, 0.5))
scene.zoom(3)

# display
# fury.window.show(scene, size=(600, 600), reset_camera=False)
fury.window.record(scene, out_path="cube.png", size=(600, 600))
