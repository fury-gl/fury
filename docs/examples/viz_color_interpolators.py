"""
============================
Keyframe Color Interpolators
============================

Color animation explained in this tutorial and how to use different color
space interpolators.
"""

import numpy as np

from fury import actor, window
from fury.animation import Animation
from fury.animation.interpolator import (
    hsv_color_interpolator,
    lab_color_interpolator,
    step_interpolator,
    xyz_color_interpolator,
)
from fury.animation.timeline import Timeline
from fury.colormap import distinguishable_colormap

scene = window.Scene()

showm = window.ShowManager(
    scene=scene, size=(900, 768), reset_camera=False, order_transparent=True
)


###############################################################################
# Initializing positions of the cubes that will be color-animated.
cubes_pos = np.array(
    [
        [[-2, 0, 0]],
        [[0, 0, 0]],
        [[2, 0, 0]],
        [[4, 0, 0]],
        [[6, 0, 0]],
    ]
)

###############################################################################
# Static labels for different interpolators (for show)
linear_text = actor.vector_text(text="Linear", pos=(-2.64, -1, 0))
lab_text = actor.vector_text(text="LAB", pos=(-0.37, -1, 0))
hsv_text = actor.vector_text(text="HSV", pos=(1.68, -1, 0))
xyz_text = actor.vector_text(text="XYZ", pos=(3.6, -1, 0))
step_text = actor.vector_text(text="Step", pos=(5.7, -1, 0))
scene.add(step_text, lab_text, linear_text, hsv_text, xyz_text)

###############################################################################
# Creating an animation to animate the actor.
# Also cube actor is provided for each timeline to handle as follows:
# ``Animation(actor)``, ``Animation(list_of_actors)``, or actors can be added
# later using ``animation.add()`` or ``animation.add_actor()``
anim_linear_color = Animation(actors=actor.cube(cubes_pos[0]))
anim_LAB_color = Animation(actors=actor.cube(cubes_pos[1]))
anim_HSV_color = Animation(actors=actor.cube(cubes_pos[2]))
anim_XYZ_color = Animation(actors=actor.cube(cubes_pos[3]))
anim_step_color = Animation(actors=actor.cube(cubes_pos[4]))

###############################################################################
# Creating a timeline to control all the animations (one for each color
# interpolation method)

timeline = Timeline(playback_panel=True)

###############################################################################
# Adding animations to a Timeline.
timeline.add_animation(
    [anim_linear_color, anim_LAB_color, anim_HSV_color, anim_XYZ_color, anim_step_color]
)

###############################################################################
# Setting color keyframes
# =======================
#
# Setting the same color keyframes to all the animations

###############################################################################
# First, we generate some distinguishable colors
colors = distinguishable_colormap(nb_colors=4)

###############################################################################
# Then, we set them as keyframes for the animations
for t in range(0, 20, 5):
    col = colors.pop()
    anim_linear_color.set_color(t, col)
    anim_LAB_color.set_color(t, col)
    anim_HSV_color.set_color(t, col)
    anim_XYZ_color.set_color(t, col)
    anim_step_color.set_color(t, col)

###############################################################################
# Changing the default scale interpolator to be a step interpolator
# The default is linear interpolator for color keyframes
anim_HSV_color.set_color_interpolator(hsv_color_interpolator)
anim_LAB_color.set_color_interpolator(lab_color_interpolator)
anim_step_color.set_color_interpolator(step_interpolator)
anim_XYZ_color.set_color_interpolator(xyz_color_interpolator)

###############################################################################
# Adding the main timeline to the show manager
showm.add_animation(timeline)

interactive = False

if interactive:
    showm.start()

window.record(scene=scene,
              out_path="viz_keyframe_animation_colors.png",
              size=(900, 768))
