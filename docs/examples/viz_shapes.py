# -*- coding: utf-8 -*-
"""
==============
Simple Shapes
==============

This example shows how to use the UI API. We will demonstrate how to draw
some geometric shapes from FURY UI elements.

First, a bunch of imports.
"""

import fury

##############################################################################
# First we need to fetch some icons that are included in FURY.

fury.data.fetch_viz_icons()

###############################################################################
# Let's draw some simple shapes. First, a rectangle.

rect = fury.ui.Rectangle2D(size=(100, 100), position=(400, 400), color=(1, 0, 1))

###############################################################################
# Then we can draw a solid circle, or disk.

disk = fury.ui.Disk2D(outer_radius=50, center=(400, 200), color=(1, 1, 0))

###############################################################################
# Add an inner radius to make a ring.

ring = fury.ui.Disk2D(
    outer_radius=50, inner_radius=45, center=(500, 600), color=(0, 1, 1)
)


###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = fury.window.ShowManager(size=current_size, title="FURY Shapes Example")

show_manager.scene.add(rect)
show_manager.scene.add(disk)
show_manager.scene.add(ring)

interactive = False

if interactive:
    show_manager.start()

fury.window.record(scene=show_manager.scene,
                   size=current_size,
                   out_path="viz_shapes.png")
