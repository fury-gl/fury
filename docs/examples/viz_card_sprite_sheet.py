# -*- coding: utf-8 -*-
"""
====
Card
====

This example shows how to create a card and use a sprite
sheet to update the image in the card.

First, some imports.
"""
import os
from tempfile import TemporaryDirectory as InTemporaryDirectory

from fury import ui, window
from fury.data import fetch_viz_icons
from fury.io import load_image, load_sprite_sheet, save_image

##############################################################################
# First we need to fetch some icons that are included in FURY.

TARGET_FPS = 15
FRAME_TIME = (1.0 / TARGET_FPS) * 1000

fetch_viz_icons()

sprite_sheet = load_sprite_sheet('https://raw.githubusercontent.com/fury-gl/'
                                 'fury-data/master/unittests/fury_sprite.png',
                                 5, 5)
CURRENT_SPRITE_IDX = 0

vtk_sprites = []
###############################################################################
# Let's create a card and add it to the show manager

img_url = "https://raw.githubusercontent.com/fury-gl"\
          "/fury-communication-assets/main/fury-logo.png"

title = "FURY"
body = "FURY - Free Unified Rendering in pYthon."\
       "A software library for scientific visualization in Python."

card = ui.elements.Card2D(image_path=img_url, title_text=title,
                          body_text=body,
                          image_scale=0.55, size=(300, 300),
                          bg_color=(1, 0.294, 0.180),
                          bg_opacity=0.8, border_width=5,
                          border_color=(0.1, 0.4, 0.8))

###############################################################################
# Now we define the callback to update the image on card after some delay.


def timer_callback(_obj, _evt):
    global CURRENT_SPRITE_IDX, show_manager
    CURRENT_SPRITE_IDX += 1
    sprite = vtk_sprites[CURRENT_SPRITE_IDX % len(vtk_sprites)]
    card.image.set_img(sprite)
    i_ren = show_manager.scene.GetRenderWindow()\
        .GetInteractor().GetInteractorStyle()

    i_ren.force_render()

###############################################################################
# Lets create a function to convert the sprite to vtkImageData


def sprite_to_vtk():
    with InTemporaryDirectory() as tdir:
        for idx, sprite in enumerate(list(sprite_sheet.values())):
            sprite_path = os.path.join(tdir, f'{idx}.png')
            save_image(sprite, sprite_path, compression_quality=100)
            vtk_sprite = load_image(sprite_path, as_vtktype=True)
            vtk_sprites.append(vtk_sprite)


###############################################################################
# Now that the card has been initialised, we add it to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Card Example")

show_manager.scene.add(card)
show_manager.initialize()
###############################################################################
# Converting numpy array sprites to vtk images
sprite_to_vtk()

###############################################################################
# Adding a timer to update the card image
show_manager.add_timer_callback(True, int(FRAME_TIME), timer_callback)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, out_path="card_ui.png", size=(1000, 1000))
