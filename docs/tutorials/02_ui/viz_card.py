# -*- coding: utf-8 -*-
"""
===============
Buttons & Text
===============

This example shows how to use the UI API. We will demonstrate how to create
panel having multiple cards on them.

First, some imports.
"""
from numpy.core.fromnumeric import size
from fury import ui, window
from fury.data import read_viz_icons, fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create some cards and put them in a panel.
# First we'll make the panel.

panel = ui.Panel2D(size=(900, 800), color=(1.0, 1.0, 1.0))
panel.center = (500, 430)

###############################################################################
# Then we'll create two cards and add them to the panel.
#
# Note that here we specify the positions with floats. In this case, these are
# percentages of the panel size.

first_img_path = read_viz_icons(fname="circle-left.png")
second_img_path = read_viz_icons(fname="circle-right.png")

# Now we create two cards with these images , with image scale set to 0.5

first_card = ui.Card2D(image_path=first_img_path, title="First Card",
                       body="First Card Body", image_scale=0.5,
                       size=(400, 400))

second_card = ui.Card2D(image_path=second_img_path, title="Second Card",
                        body="Second Card Body", image_scale=0.5,
                        size=(400, 400))

panel.add_element(first_card, (0, 0.25))
panel.add_element(second_card, (0.556, 0.25))

###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Card Example")

show_manager.scene.add(panel)
show_manager.start()
