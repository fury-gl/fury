"""
========================================
Sphere Color Control using Radio Buttons
========================================

This example shows how to use the UI API. We will demonstrate how to
create a Sphere and control its color using radio buttons.

First, some imports.
"""

import numpy as np

import fury

##############################################################################
# First we need to fetch some icons that are included in FURY.

fury.data.fetch_viz_icons()

########################################################################
# Sphere and Radio Buttons
# ========================
#
# Add a Sphere to the scene.

sphere = fury.actor.sphere(
    centers=np.array([[50, 0, 0]]),
    colors=np.array([[0, 0, 1]]),
    radii=11.0,
    theta=360,
    phi=360,
)

# Creating a dict of possible options and mapping it with their values.
options = {"Blue": (0, 0, 255), "Red": (255, 0, 0), "Green": (0, 255, 0)}

color_toggler = fury.ui.RadioButton(
    list(options),
    checked_labels=["Blue"],
    padding=1,
    font_size=16,
    font_family="Arial",
    position=(200, 200),
)


# A callback which will set the values for the box
def toggle_color(radio):
    vcolors = fury.utils.colors_from_actor(sphere)
    color = options[radio.checked_labels[0]]
    vcolors[:] = np.array(color)
    fury.utils.update_actor(sphere)


color_toggler.on_change = toggle_color


###############################################################################
# Show Manager
# ============
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = fury.window.ShowManager(size=current_size, title="FURY Sphere Example")

show_manager.scene.add(sphere)
show_manager.scene.add(color_toggler)

###############################################################################
# Set camera for better visualization

show_manager.scene.reset_camera()
show_manager.scene.set_camera(position=(0, 0, 150))
show_manager.scene.reset_clipping_range()
show_manager.scene.azimuth(30)
interactive = False

if interactive:
    show_manager.start()

fury.window.record(
    show_manager.scene, size=current_size, out_path="viz_radio_buttons.png"
)
