# -*- coding: utf-8 -*-
"""
===========
SpinBox UI
===========

This example shows how to use the UI API. We will demonstrate how to create
a SpinBox UI.

First, some imports.
"""
from fury import ui, window
from fury.data import read_viz_icons, fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create some buttons and text and put them in a panel.
# First we'll make the panel.

spinbox = ui.SpinBox()

###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY SpinBox Example")

show_manager.scene.add(spinbox)

interactive = True

if interactive:
    show_manager.start()

# window.record(show_manager.scene, size=current_size,
#               out_path="viz_spinbox.png")
