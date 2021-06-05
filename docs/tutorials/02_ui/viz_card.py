# -*- coding: utf-8 -*-
"""
===============
Card
===============

This example shows how to create a card.

First, some imports.
"""
from fury import ui, window
from fury.data import read_viz_icons, fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create a card and add it to the show manager

img_path = read_viz_icons(fname="stop2.png")

card = ui.Card2D(image_path=img_path, title_text="Card Title",
                 body_text="This is the body.\nProvides addition info",
                 image_scale=0.5, size=(400, 400),
                 bg_color=(0.498, 0.584, 0.819),
                 bg_opacity=0.8)


###############################################################################
# Now that the card has been initialised, we add it to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Card Example")

show_manager.scene.add(card)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, out_path="card_ui.png", size=(1000, 1000))
