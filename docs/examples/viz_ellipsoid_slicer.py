"""
=======================
Ellipsoid Field Slicing
=======================

Visualize one Z slice of an ellipsoid field. Press Up/Down to move the slice.
Across Z, the longest axes rotate from X (red) to Z (blue).
"""

import numpy as np

from fury import actor, window

###############################################################################
# Define an 8x8x8 field whose first semi-axis grows along Z.

lengths = np.empty((8, 8, 8, 3), dtype=np.float32)
lengths[..., 0] = np.linspace(0.15, 0.4, 8)
lengths[..., 1:] = 0.15

# Each matrix's columns define the glyph axes. Rotate the longest axis from X to Z.
rotations = np.empty((8, 3, 3), dtype=np.float32)
for z, angle in enumerate(np.linspace(0, np.pi / 2, 8)):
    cosine, sine = np.cos(angle), np.sin(angle)
    rotations[z] = [[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]]
orientations = np.broadcast_to(rotations, (*lengths.shape[:3], 3, 3))

###############################################################################
# Show the central Z plane. Scale only the glyphs; default colors follow their
# longest axes, not the slice position or later world transforms.

slicer = actor.ellipsoid_slicer(
    lengths,
    orientation_matrices=orientations,
    scale=0.8,
    visibility=(False, False, True),
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
