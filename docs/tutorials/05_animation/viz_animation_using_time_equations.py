"""
=====================
Keyframe animation
=====================

Tutorial on making keyframe-based animation in FURY using custom functions.

"""
from cmath import sin, cos

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


cube = actor.cube(np.array([[0, 0, 0]]), (0, 0, 0), (1, 0, 1), scales=6)

###############################################################################
# Creating a timeline to animate the actor
timeline = Timeline(playback_panel=True, length=2 * np.pi, loop=True)

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
timeline.add_actor(cube)


###############################################################################
# Creating time dependent functions.
def pos_eval(t):
    return np.array([np.sin(t), np.cos(t) * np.sin(t), 0]) * 15


def color_eval(t):
    return (np.array([np.sin(t), np.sin(t - 2 * np.pi / 3),
                      np.sin(t + np.pi / 3)]) + np.ones(3)) / 2


def rotation_eval(t):
    return np.array([np.sin(t) * 360, np.cos(t) * 360, 0])


def scale_eval(t):
    return (np.array([np.sin(t), np.sin(t - 2 * np.pi / 3),
                      np.sin(t + np.pi / 3)]) + np.ones(3) * 2) / 5


###############################################################################
# Setting evaluator functions is the same as setting interpolators, but with
# one extra argument: `is_evaluator=True` since these functions does not need
# keyframes as input.
timeline.set_position_interpolator(pos_eval, is_evaluator=True)
timeline.set_rotation_interpolator(rotation_eval, is_evaluator=True)
timeline.set_color_interpolator(color_eval, is_evaluator=True)
timeline.set_interpolator('scale', scale_eval, is_evaluator=True)

###############################################################################
# Main timeline to control all the timelines.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding timelines to the main Timeline.
scene.add(timeline)


###############################################################################
# making a function to update the animation and render the scene.
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation.
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_evaluators.png',
              size=(900, 768))
