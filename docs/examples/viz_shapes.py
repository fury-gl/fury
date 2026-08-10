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

from fury.ui import Rectangle2D, Disk2D, RoundedRectangle2D
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
# And a rounded rectangle with custom corner radius and opacity.

rounded_rect = RoundedRectangle2D(
    size=(150, 100),
    position=(500, 200),
    color=(0.2, 0.6, 0.9),
    corner_radius=20,
    opacity=0.9,
)

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


_rounded_rect_drag_offset = None


def rounded_rect_left_button_pressed(event: PointerEvent):
    """Calculates offset when dragging starts."""
    global _rounded_rect_drag_offset

    rect_position = rounded_rect.get_position()
    click_pos = np.array([event.x, event.y])
    _rounded_rect_drag_offset = click_pos - rect_position

    rounded_rect.color = np.random.random([3])


def rounded_rect_left_button_dragged(event: PointerEvent):
    """Updates the rounded rectangle's position based on mouse movement."""
    global _rounded_rect_drag_offset
    if _rounded_rect_drag_offset is not None:
        click_position = np.array([event.x, event.y])
        new_position = click_position - _rounded_rect_drag_offset
        rounded_rect.set_position(new_position)


def rounded_rect_left_button_released(event: PointerEvent):
    """Resets the drag offset."""
    global _rounded_rect_drag_offset
    _rounded_rect_drag_offset = None


rounded_rect.on_left_mouse_button_pressed = rounded_rect_left_button_pressed
rounded_rect.on_left_mouse_button_dragged = rounded_rect_left_button_dragged
rounded_rect.on_left_mouse_button_released = rounded_rect_left_button_released


###############################################################################
# Now that all the elements have been initialised, we add them to the scene.
scene.add(rect)
scene.add(disk)
scene.add(rounded_rect)

current_size = (800, 800)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY Shapes Example",
)
show_manager.start()
