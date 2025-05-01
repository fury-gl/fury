"""
===============
3D Image Slicer
===============

This example demonstrates how to use the `slicer` actor to visualize
3D data in a 2D slice view. The `slicer` actor allows you to interactively
slice through a 3D volume and visualize the resulting 2D slices.

we will first create a 3D volume slicer from DIPY's MNI template
and then add event handlers to allow the user to interact with the slices.
The user can pick a voxel in the slice to get its intensity value
and use arrow keys to navigate through the slices.

"""

import numpy as np
from fury import actor, window
from fury.utils import get_slices, show_slices
from dipy.data import read_mni_template


###############################################################################
# Let's read the 3D data from dipy and check the shape of the data.

nifti = read_mni_template()
data = np.asarray(nifti.dataobj)
print(data.shape)

###############################################################################
# Create slice actor to visualize the 3D data as XY, YZ and XZ slices.

slicer_actor = actor.slicer(data)

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

    position = np.maximum(np.zeros((3,)), position)
    position = np.minimum(np.asarray(data.shape), position)
    show_slices(slicer_actor, position)
    show_m.render()


###############################################################################
# Add event handlers to the slicer actor for picking and key events.

slicer_actor.add_event_handler(handle_pick_event, "pointer_down")

show_m = window.ShowManager(scene=scene, title="FURY 2.0: MNI Template Slicer Example")
show_m.renderer.add_event_handler(handle_key_event, "key_down")

if __name__ == "__main__":
    show_m.start()


###############################################################################
# Create a 3D cube with a sphere inside it.
# This example demonstrates how to create a 3D cube with a sphere inside it
# and visualize it using the `slicer` actor. The cube is filled with blue color,
# and the sphere is filled with white color. The user can interactively slice
# through the cube and visualize the resulting 2D slices.

size = 100
cube = np.zeros((size, size, size), dtype=np.float32)

# Create coordinate grids
x, y, z = np.indices((size, size, size))
center = (size - 1) / 2

# Calculate distances from center
distances = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

# Create sphere (white)
sphere_mask = distances <= size // 2
cube[sphere_mask] = 1.0

# Create cube (blue)
cube[~sphere_mask] = 0.5

################################################################################
# Create a slice actor for the cube and add it to the scene.
slicer_actor = actor.slicer(cube)
scene = window.Scene()
scene.add(slicer_actor)

################################################################################
# Add event handlers to the slice actor for key events.

show_m = window.ShowManager(scene=scene, title="FURY 2.0: Cube Slicer Example")
show_m.renderer.add_event_handler(handle_key_event, "key_down")

if __name__ == "__main__":
    show_m.start()
