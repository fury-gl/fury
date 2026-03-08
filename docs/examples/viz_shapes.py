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
import numpy as np

from fury.ui import Rectangle2D, Disk2D
from fury.window import (
    Scene,
    ShowManager,
)
from fury.lib import PointerEvent

##############################################################################
# Creating a Scene

scene = Scene()

###############################################################################
# Let's draw some simple shapes. First, a rectangle.

rect = Rectangle2D(size=(100, 100), position=(400, 400), color=(1, 0, 1))

###############################################################################
# Then we can draw a solid circle, or disk.

disk = Disk2D(outer_radius=50, center=(200, 200), color=(1, 1, 0))

###############################################################################
# Adding interaction using mouse events


_rect_drag_offset = None


def rect_left_button_pressed(event: PointerEvent):
    """Calculates offset when dragging starts."""
    global _rect_drag_offset

    rect_position = rect.get_position()
    click_pos = np.array([event.x, event.y])
    _rect_drag_offset = click_pos - rect_position


def rect_left_button_dragged(event: PointerEvent):
    """Updates the rectangle's position based on mouse movement."""
    global _rect_drag_offset
    if _rect_drag_offset is not None:
        click_position = np.array([event.x, event.y])
        new_position = click_position - _rect_drag_offset
        rect.set_position(new_position)


rect.on_left_mouse_button_pressed = rect_left_button_pressed
rect.on_left_mouse_button_dragged = rect_left_button_dragged


def disk_right_button_pressed(event: PointerEvent):
    """Changes the disk's color to a random RGB value on press."""
    disk.color = np.random.random([3])


disk.on_right_mouse_button_pressed = disk_right_button_pressed


###############################################################################
# Now that all the elements have been initialised, we add them to the scene.
scene.add(rect)
scene.add(disk)

current_size = (800, 800)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY Shapes Example",
)
show_manager.start()
