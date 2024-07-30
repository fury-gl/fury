"""
=====================
Keyframe animation
=====================

Tutorial on making keyframe-based animation in FURY using custom functions.
"""

import numpy as np

import fury

scene = fury.window.Scene()

showm = fury.window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)


cube = fury.actor.cube(np.array([[0, 0, 0]]), (0, 0, 0), (1, 0, 1), scales=6)

###############################################################################
# Creating an ``Animation`` to animate the actor and show its motion path.
anim = fury.animation.Animation(length=2 * np.pi, loop=True, motion_path_res=200)

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
anim.add_actor(cube)


###############################################################################
# Creating time dependent functions.
def pos_eval(t):
    return np.array([np.sin(t), np.cos(t) * np.sin(t), 0]) * 15


def color_eval(t):
    return (
        np.array([np.sin(t), np.sin(t - 2 * np.pi / 3), np.sin(t + np.pi / 3)])
        + np.ones(3)
    ) / 2


def rotation_eval(t):
    return np.array([np.sin(t) * 360, np.cos(t) * 360, 0])


def scale_eval(t):
    return (
        np.array([np.sin(t), np.sin(t - 2 * np.pi / 3), np.sin(t + np.pi / 3)])
        + np.ones(3) * 2
    ) / 5


###############################################################################
# Setting evaluator functions is the same as setting interpolators, but with
# one extra argument: `is_evaluator=True` since these functions does not need
# keyframes as input.
anim.set_position_interpolator(pos_eval, is_evaluator=True)
anim.set_rotation_interpolator(rotation_eval, is_evaluator=True)
anim.set_color_interpolator(color_eval, is_evaluator=True)
anim.set_interpolator("scale", scale_eval, is_evaluator=True)

###############################################################################
# changing camera position to observe the animation better.
scene.set_camera(position=(0, 0, 90))

###############################################################################
# Adding the animation to the show manager.
showm.add_animation(anim)


interactive = False

if interactive:
    showm.start()

fury.window.record(
    scene, out_path="viz_keyframe_animation_evaluators.png", size=(900, 768)
)
