# -*- coding: utf-8 -*-
"""
==============
Simple Shapes
==============

This example shows how to use the UI API. We will demonstrate how to draw
some geometric shapes from FURY UI elements.
"""

##############################################################################
# First, a bunch of imports

from fury.ui import Rectangle2D, Disk2D, UIContext
from fury.window import (
    Scene,
    ShowManager,
)

##############################################################################
# Using UI v2 Version

UIContext.set_is_v2_ui(True)

##############################################################################
# Creating a Scene

scene = Scene()

###############################################################################
# Let's draw some simple shapes. First, a rectangle.

rect = Rectangle2D(size=(100, 100), position=(400, 400), color=(1, 0, 1))

###############################################################################
# Then we can draw a solid circle, or disk.

disk = Disk2D(outer_radius=50, center=(400, 200), color=(1, 1, 0))

###############################################################################
# Now that all the elements have been initialised, we add them to the scene.
scene.add(rect)
scene.add(disk)

if __name__ == "__main__":
    current_size = (800, 800)
    show_manager = ShowManager(
        scene=scene, size=current_size, title="FURY 2.0: Shapes Example"
    )
    show_manager.start()
