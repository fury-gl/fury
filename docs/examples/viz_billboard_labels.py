# -*- coding: utf-8 -*-
"""
=================================
Billboard labels actor behaviosrs
=================================

This examples shows some functionalities of the `billboard_labels` actor.
We show here how to change the font-size (resolution), the font face, alignment
and offsets.

If you want to see the labels with a different font, you need to install
the Freetype library and freetype-py (`pip install freetype-py`).

"""

###############################################################################
# First, let's import some useful functions
import fury
from fury import actor, window
from fury import text_tools
import numpy as np

# Set to True to enable user interaction
interactive = True

# Create a window FURY
scene = window.Scene()
# scene.background((1., 1., 1.))
centers = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 0],
])
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [.15, .5, .75],
    [.75, .15, .5]
])


sphere_actor = actor.sphere(
    centers=centers,
    colors=colors,
    radii=.05,
)

scene.add(sphere_actor)

# Create a text actor aligned to the center of the first sphere
center_actor = actor.bitmap_labels(
    [centers[0]], ['Center align'],
    border_width=0.1,
    border_color=(.75, .25, 0, .5),
    border_type='solid',
    font_name='FreeMono',
    align='center', scales=.08, colors=colors[0])

scene.add(center_actor)

# Create a text actor aligned to the right of the second sphere
right_actor = actor.bitmap_labels(
    [centers[1]], ['Right align'],
    align='right', scales=.05, colors=colors[1])

scene.add(right_actor)

# Create a text actor aligned to the left of the third sphere
left_actor = actor.bitmap_labels(
    [centers[2]], ['Left align'],
    align='left', scales=.05, colors=colors[2])

scene.add(left_actor)

##############################################################################
# To list all the available fonts, run:
print('\n\t Fonts available: ', text_tools.list_fonts_available(), '\n')
# We will use the InconsolataBold700 font in this label actor
#
offset_change_actor = actor.bitmap_labels(
    [centers[3]], ['Inconsolata'],
    font_name='InconsolataBold700',
    scales=.05, colors=colors[3])

scene.add(offset_change_actor)
###############################################################################
# We can change the offsets of the labels with the `x_offset_ratio` and
# `y_offset_ratio` arguments. The offset is computed as a ratio of the
# font size.
#
offset_change_actor = actor.bitmap_labels(
    [centers[4]], ['offset'],
    x_offset_ratio=3, y_offset_ratio=2,
    scales=.05, colors=colors[4])

scene.add(offset_change_actor)

###############################################################################
# We can change the border width and color of the labels with the
# `border_width` and `border_color` arguments. 


text_with_border = actor.bitmap_labels(
    [centers[5]], ['Text with a border'],
    border_width=.1, border_color=(.75, .25, 0, .5),
    border_type='solid',
    align='center', scales=0.1, colors=colors[5])

scene.add(text_with_border)

scene.reset_camera()
scene.reset_clipping_range()


if interactive:
    window.show(scene, size=(600, 600), order_transparent=True)

window.record(scene, out_path='viz_billboard_labels.png', size=(600, 600))
