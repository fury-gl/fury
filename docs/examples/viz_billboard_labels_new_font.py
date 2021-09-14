# -*- coding: utf-8 -*-
"""
=================================
Billboard labels new font example
=================================

If you want to see the labels with a different font, you need to install
the Freetype library (`pip install freetype-py`).

"""

###############################################################################
# First, let's import some useful functions
import fury
from fury import actor, window
from fury import text_tools
import numpy as np

# Set to True to enable user interaction
interactive = False

# Create a window FURY
scene = window.Scene()


###############################################################################
# To create a label with a different font, you need to have the FreeType
# library and freetype-py installed.
#
if text_tools._FREETYPE_AVAILABLE:

    # We start choosing a path for the TTF file. Here we use the Roboto font
    # that is available on the FURY examples folder.
    font_path = f'{fury.__path__[0]}/data/files/RobotoMonoBold700.ttf'

###############################################################################
# Then we create the texture atlas for the font. The `font_size_res`
# argument controls the quality of the font rendering, the higher the better
#
    text = 'A custom font with special characters like: ç, ã and à'
    # # The `label` need to have special characters thus we will tell the
    # # `create_atlas_font` to draw those characters.
    chars = list(set(text))
    text_tools.create_new_font(
        'FreeMonoWithSpecial', font_path=font_path, font_size_res=10,
        chars=chars, force_recreate=True)
    new_font_actor = actor.bitmap_labels(
        [np.array([0., 0., 0.])], [text],
        font_name='FreeMonoWithSpecial',
        align='center', scales=0.1,)

    scene.add(new_font_actor)

scene.reset_camera()
scene.reset_clipping_range()


if interactive:
    window.show(scene, size=(600, 600))

window.record(scene, out_path='viz_billboard_labels.png', size=(600, 600))
