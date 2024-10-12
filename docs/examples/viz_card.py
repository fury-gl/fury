# -*- coding: utf-8 -*-
"""
====
Card
====

This example shows how to create a card.

First, some imports.
"""

import fury
from fury.data import fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

###############################################################################
# Let's create a card and add it to the show manager

img_url = (
    "https://raw.githubusercontent.com/fury-gl"
    "/fury-communication-assets/main/fury-logo.png"
)

title = "FURY"
body = (
    "FURY - Free Unified Rendering in pYthon."
    "A software library for scientific visualization in Python."
)

card = fury.ui.elements.Card2D(
    image_path=img_url,
    title_text=title,
    body_text=body,
    image_scale=0.55,
    size=(300, 300),
    bg_color=(1, 0.294, 0.180),
    bg_opacity=0.8,
    border_width=5,
    border_color=(0.1, 0.4, 0.4),
)

###############################################################################
# Now that the card has been initialised, we add it to the show
# manager.

current_size = (1000, 1000)
show_manager = fury.window.ShowManager(size=current_size, title="FURY Card Example")

show_manager.scene.add(card)
# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

fury.window.record(scene=show_manager.scene, out_path="card_ui.png", size=(1000, 1000))
