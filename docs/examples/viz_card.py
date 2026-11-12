"""
======
Card2D
======

This example demonstrates the ``Card2D`` UI element. A card stacks an image, a
title and a body text on a background panel, with an optional row of icon
buttons at the bottom, like the cards of many websites with a heart to like and
a bookmark to save. Cards can be dragged around the window with the mouse
unless ``draggable`` is set to False.

The card images are the sky pictures of FURY's skybox dataset. The action icons
come from the Lucide pack. Both are downloaded the first time they are needed.
"""

##############################################################################
# First, let's import the necessary modules.

from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.ui import Card2D, TextBlock2D
from fury.window import Scene, ShowManager

##############################################################################
# Fetch the skybox pictures to use as card images. ``read_viz_cubemap`` returns
# the paths of the six faces of the skybox.

fetch_viz_cubemaps()
sun, golden_sky, _, storm, _, _ = read_viz_cubemap("skybox")

##############################################################################
# Create a Scene with a soft background.

scene = Scene(background="#F1F3FB")

##############################################################################
# **Card 1 - Actions.**
#
# ``actions`` takes Lucide icon names. Each one becomes a toggle button in the
# ``actions`` dictionary of the card. A toggled button is highlighted, which
# suits a like or a save. A share button is not a toggle, so we turn that off.
# The colors accept a hex string or RGB values in [0, 1] or [0, 255].

card_1 = Card2D(
    golden_sky,
    title_text="Above the Clouds",
    body_text="A calm evening sky with a golden horizon.",
    position=(30, 50),
    size=(220, 400),
    bg_color="#E7E5FF",
    title_color="#2B2A4C",
    body_color="#5B5A7A",
    actions=["heart", "bookmark", "share-2"],
)
card_1.actions["share-2"].is_toggle = False

##############################################################################
# ``on_clicked`` is called with the button each time it is clicked. We use it
# to report the state of the heart in a ``TextBlock2D``.

status = TextBlock2D(
    text="Click the heart.",
    font_size=18,
    color="#2B2A4C",
    position=(30, 20),
    dynamic_bbox=True,
)


def show_like(button):
    """
    Report whether the card is liked.

    Parameters
    ----------
    button : TexturedButton2D
        The clicked heart button.
    """
    status.message = "Liked Above the Clouds" if button.toggled else "Like removed"


card_1.actions["heart"].on_clicked = show_like

##############################################################################
# **Card 2 - A different look.**
#
# ``border_color`` and ``border_width`` (in pixels) draw a frame around the card.
# ``image_scale`` is the fraction of the card height taken by the image and
# ``padding`` the space in pixels between the elements.

card_2 = Card2D(
    sun,
    title_text="Golden Hour",
    body_text="Warm tones with a thin frame around the card.",
    position=(280, 50),
    size=(220, 400),
    image_scale=0.55,
    padding=16,
    bg_color="#FFF1E6",
    title_color="#7A3E00",
    body_color="#9A6B3D",
    border_color="#FFB066",
    border_width=3,
    actions=["heart", "message-circle"],
)
card_2.actions["message-circle"].is_toggle = False

##############################################################################
# **Card 3 - A locked card.**
#
# Setting ``draggable=False`` locks the card in place. The texts of a card can
# be changed after it is created.

card_3 = Card2D(
    storm,
    title_text="Eye of the Storm",
    body_text="This card cannot be dragged.",
    position=(530, 50),
    size=(220, 400),
    bg_color="#E8EEF4",
    title_color="#1F3A52",
    body_color="#4C6A85",
    draggable=False,
)
card_3.body = "This card cannot be dragged (draggable=False)."

##############################################################################
# Add everything to the scene.

scene.add(card_1, card_2, card_3, status)

##############################################################################
# Create and start the ShowManager.

show_manager = ShowManager(
    scene=scene,
    size=(780, 480),
    title="FURY Card2D Example",
)
show_manager.start()
