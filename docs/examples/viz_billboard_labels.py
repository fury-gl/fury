"""
=================================
Billboard labels actor behaviosrs
=================================

This examples shows some functionalities of the `billboard_labels` actor.
We show here how to change the font-size (resolution), the font face, alignment
and offsets.

"""

###############################################################################
# First, let's import some useful functions
import fury
from fury import actor, window
import numpy as np


scene = window.Scene()
centers = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
])
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [.15, .5, .75],
    [.75, .15, .5]
])

sphere_actor = actor.sphere(
    centers=centers,
    colors=colors,
    radii=.05,
)

scene.add(sphere_actor)

center_actor = actor.bitmap_labels(
    [centers[0]], ['Center align'],
    align='center', scales=.05, colors=colors[0])

scene.add(center_actor)

right_actor = actor.bitmap_labels(
    [centers[1]], ['Right align'],
    align='right', scales=.05, colors=colors[1])

scene.add(right_actor)

left_actor = actor.bitmap_labels(
    [centers[2]], ['Left align'],
    align='left', scales=.05, colors=colors[2])

scene.add(left_actor)

font_path = font_path = f'{fury.__path__[0]}/data/files/verdanab.ttf'
new_font_actor = actor.bitmap_labels(
    [centers[3]], ['font size and format'],
    font_size=10, font_path=font_path,
    align='center', scales=.05, colors=colors[3])

offset_change_actor = actor.bitmap_labels(
    [centers[4]], ['offset'],
    x_offset_ratio=3, y_offset_ratio=2,
    scales=.05, colors=colors[4])

scene.add(offset_change_actor)
scene.add(new_font_actor)
scene.reset_camera()
scene.reset_clipping_range()

interactive = False

if interactive:
    window.show(scene, size=(600, 600))

window.record(scene, out_path='viz_billboard_labels.png', size=(600, 600))
