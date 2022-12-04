"""
=====================
Keyframe animation
=====================

Minimal tutorial of making keyframe-based animation in FURY.

"""

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import cubic_spline_interpolator

scene = window.Scene()

showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)


arrow = actor.arrow(np.array([[0, 0, 0]]), (0, 0, 0), (1, 0, 1), scales=6)

###############################################################################
# Creating a timeline to animate the actor
timeline = Timeline(playback_panel=True)

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
timeline.add_actor(arrow)

###############################################################################
# Adding some position keyframes
timeline.set_position(0, np.array([0, 0, 0]))
timeline.set_position(2, np.array([10, 10, 10]))
timeline.set_position(5, np.array([-10, 16, 0]))
timeline.set_position(9, np.array([10, 0, 20]))

###############################################################################
# change the position interpolator to Cubic spline interpolator.
timeline.set_position_interpolator(cubic_spline_interpolator)

###############################################################################
# Adding some rotation keyframes.
timeline.set_rotation(0, np.array([160, 50, 20]))
timeline.set_rotation(4, np.array([60, 160, 0]))
timeline.set_rotation(8, np.array([0, -180, 90]))

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

window.record(scene, out_path='viz_keyframe_animation_introduction.png',
              size=(900, 768))
