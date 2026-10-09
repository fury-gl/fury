"""
=======================
Ellipsoid Field Slicing
=======================

Visualize one Z slice of an ellipsoid field. Press Up/Down to move the slice.
Each voxel stores three semi-axis lengths; orientations default to X, Y, Z.
"""

import numpy as np

from fury import actor, window

###############################################################################
# Define an 8x8x8 field. The X semi-axis grows along Z to make slice changes visible.

lengths = np.empty((8, 8, 8, 3), dtype=np.float32)
lengths[..., 0] = np.linspace(0.15, 0.4, 8)
lengths[..., 1:] = 0.15

###############################################################################
# Display only the Z plane, initially at the center of the field.

slicer = actor.ellipsoid_slicer(
    lengths, colors=(0.0, 0.7, 1.0), visibility=(False, False, True)
)
scene = window.Scene()
scene.add(slicer)
show_manager = window.ShowManager(
    scene=scene, size=(800, 600), title="Ellipsoid Slicer (Up, Down - Z slice)"
)

###############################################################################
# Use the same slice controls as the vector-field tutorial. Positions are clamped
# to the field bounds; other keys leave the slice unchanged.


def handle_key_event(event):
    if event.key not in ("ArrowUp", "ArrowDown"):
        return
    x, y, z = actor.get_slices(slicer)
    step = 1 if event.key == "ArrowUp" else -1
    actor.show_slices(slicer, (x, y, z + step))
    show_manager.render()


show_manager.renderer.add_event_handler(handle_key_event, "key_down")

show_manager.start()
