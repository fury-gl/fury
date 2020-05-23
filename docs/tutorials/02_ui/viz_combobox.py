# -*- coding: utf-8 -*-
"""
========
ComboBox
========

This example shows how to use the UI API. We will demonstrate how to
create ComboBoxes for selecting actors and thier respective colors.

First, some imports.
"""
from fury import ui, window, actor
import numpy as np

########################################################################
# Setting up actors.
# ===============================
#
# Adding all actors to the scene.
#
# Firstly, we define some common actor properties.

centers = np.random.rand(1, 3)
dirs = np.random.rand(1, 3)
heights = np.random.rand(5)

chosen_actor = None
chosen_color = (1, 1, 1)

########################################################################
# Now we create a cube.

cube = actor.cube(centers, dirs, (1, 1, 1), heights=heights)
cube.SetVisibility(False)

#########################################################################
# Next we create a sphere.

sphere = actor.sphere(centers, window.colors.coral)
sphere.SetVisibility(False)

########################################################################
# Next a cylinder

cylinder = actor.cylinder(centers, dirs, (1, 1, 1), heights=heights)
cylinder.SetVisibility(False)

########################################################################
# Now we create two dictionaries, one to map Actor names with their
# respective actor instances and the other to store the RGB values of
# 7 different colors.

actors = {
    "Cube": cube,
    "Sphere": sphere,
    "Cylinder": cylinder
}

colors = {
    "Violet": (0.6, 0, 0.8),
    "Indigo": (0.3, 0, 0.5),
    "Blue": (0, 0, 1),
    "Green": (0, 1, 0),
    "Yellow": (1, 1, 0),
    "Orange": (1, 0.5, 0),
    "Red": (255, 0, 0)
}

########################################################################
# ComboBox
# ===================
#
# Now we create two ComboBox UI components.
# One for selecting the actors and the other for selecting its color.

actor_combobox = ui.ComboBox2D(items=list(actors.keys()), size=(300, 200))
color_combobox = ui.ComboBox2D(items=list(colors.keys()), size=(300, 200))

########################################################################
# Callbacks
# ==================================
#
# Now we create callbacks for setting the chosen actor and color.

def change_actor(combobox):

    global chosen_actor, chosen_color

    # Identify the option using `combobox.text`
    chosen_actor = actors[combobox.text]

    for actr in combobox.items:
        if actr != combobox.text:
            actors[actr].SetVisibility(False)

    chosen_actor.SetVisibility(True)
    chosen_actor.GetProperty().SetColor(*chosen_color)

def change_color(combobox):

    global chosen_actor, chosen_color

    if chosen_actor is not None:
        chosen_color = colors[combobox.text]
        chosen_actor.GetProperty().SetColor(*chosen_color)

actor_combobox.on_change = change_actor
color_combobox.on_change = change_color

###############################################################################
# Show Manager
# ==================================
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
showm = window.ShowManager(size=current_size, title="ComboBox UI Example")
showm.scene.add(cube, sphere, cylinder, actor_combobox, color_combobox)

###############################################################################
# Set camera for better visualization

showm.scene.reset_camera()
showm.scene.reset_clipping_range()
showm.scene.azimuth(30)
interactive = False

if interactive:
    showm.start()