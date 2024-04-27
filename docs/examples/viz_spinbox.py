# -*- coding: utf-8 -*-
"""
===========
SpinBox UI
===========

This example shows how to use the UI API. We will demonstrate how to create
a SpinBox UI.

First, some imports.
"""
import numpy as np

from fury import actor, ui, utils, window
from fury.data import fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create a Cone.

cone = actor.cone(centers=np.random.rand(1, 3),
                  directions=np.random.rand(1, 3),
                  colors=(1, 1, 1), heights=np.random.rand(1))

###############################################################################
# Creating the SpinBox UI.

spinbox = ui.SpinBox(position=(200, 100), size=(300, 100), min_val=0,
                     max_val=360, initial_val=180, step=10)

###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY SpinBox Example")

show_manager.scene.add(cone)
show_manager.scene.add(spinbox)

###############################################################################
# Using the on_change hook to rotate the cone.

# Tracking previous value to check in/decrement.
previous_value = spinbox.value


def rotate_cone(spinbox):
    global previous_value
    change_in_value = spinbox.value - previous_value
    utils.rotate(cone, (change_in_value, 1, 0, 0))
    previous_value = spinbox.value


spinbox.on_change = rotate_cone

###############################################################################
# Starting the ShowManager.

interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size,
              out_path="viz_spinbox.png")
