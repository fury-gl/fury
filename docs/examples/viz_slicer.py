"""
===============
3D Image Slicer
===============

This example demonstrates how to use the `slicer` actor to visualize
3D data in a 2D slice view. The `slicer` actor allows you to interactively
slice through a 3D volume and visualize the resulting 2D slices.

we will use the slicer actor to visualize a 3D volume where we generate a cube with a
sphere inside it. The cube is filled with blue color, and the sphere is filled with
white color. We will add event handlers to allow the user
to interact with the slices

Then, we will create a 3D volume slicer from
`MNI template <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_ present
in `DIPY <https://dipy.org>`_'s data. The user can pick a voxel in the slice to get its
intensity value and use arrow keys to navigate through the slices.

"""

import numpy as np
from fury import actor, window, bio
from fury.utils import get_slices, show_slices
from dipy.data import read_mni_template

###############################################################################
# Create a 3D cube with a sphere inside it.
# This example demonstrates how to create a 3D cube with a sphere inside it
# and visualize it using the `slicer` actor. The cube is filled with blue color,
# and the sphere is filled with white color. The user can interactively slice
# through the cube and visualize the resulting 2D slices.

size = 100
cube = np.zeros((size, size, size, 3), dtype=np.float32)
cube[..., 2] = 1  # Set blue color for the cube

# Create coordinate grids
x, y, z = np.indices((size, size, size))
center = (size - 1) / 2

# Calculate distances from center
distances = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

# Create sphere (white)
sphere_mask = distances <= size // 2
cube[sphere_mask] = [1, 1, 1]

################################################################################
# Create a slice actor for the cube and add it to the scene.
slicer_actor = actor.slicer(cube)
scene = window.Scene()
scene.add(slicer_actor)

###############################################################################
# Create a function to handle key events and navigate through the slices.
# The user can use the arrow keys to move up and down through the slices.
# The `get_slices` function retrieves the current slice positions, and the
# `show_slices` function updates the displayed slices based on the new
# positions. The `show_m.render()` function is called to update the scene
# after each key event.


def handle_key_event(event):
    position = get_slices(slicer_actor)
    if event.key == "ArrowUp":
        position += 1
    elif event.key == "ArrowDown":
        position -= 1

    bounds = slicer_actor.get_bounding_box()

    position = np.maximum(bounds[0], position)
    position = np.minimum(bounds[1], position)
    show_slices(slicer_actor, position)
    show_m.render()


################################################################################
# Add event handlers to the slice actor for key events.

show_m = window.ShowManager(scene=scene, title="FURY 2.0: Cube Slicer Example")
show_m.renderer.add_event_handler(handle_key_event, "key_down")


################################################################################
# Start the show manager to display the scene and allow interaction.

show_m.start()
###############################################################################
# Let's read the 3D data from dipy and check the shape of the data. The data also has
# an affine transformation matrix that maps the voxel coordinates to world coordinates.

nifti = read_mni_template()
data = np.asarray(nifti.dataobj)
affine = nifti.affine

###############################################################################
# Let's check the shape of the data and the affine transformation matrix.
print(data.shape)
print(affine)

###############################################################################
# Create volume_slicer actor to visualize the 3D data as XY, YZ and XZ slices.

slicer_actor = bio.volume_slicer(data, affine=affine)

###############################################################################
# Create a scene and add the slicer actor to it.

scene = window.Scene()
scene.add(slicer_actor)

###############################################################################
# Create a function to handle the pick event and print the intensity of the
# voxel that was clicked. The event contains information about the picked
# voxel, including its index and intensity value.


def handle_pick_event(event):
    info = event.pick_info
    intensity = np.asarray(info["rgba"].rgb).mean()
    print(f"Voxel {info['index']}: {intensity:.2f}")


###############################################################################
# Add event handlers to the slicer actor for picking and key events.

slicer_actor.add_event_handler(handle_pick_event, "pointer_down")

show_m = window.ShowManager(scene=scene, title="FURY 2.0: MNI Template Slicer Example")
show_m.renderer.add_event_handler(handle_key_event, "key_down")

################################################################################
# Start the show manager to display the scene and allow interaction.

show_m.start()
