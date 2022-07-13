"""
=====================
Keyframe animation
=====================

Minimal tutorial of making keyframe-based animation in FURY.

"""

import numpy as np
from fury import actor, window
from fury.animation import Timeline, CubicSplineInterpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

boxes = actor.cube(np.array([[0, 0, 0]]), (1, 1, 1), (1, 0, 1))


# Creating a timeline to animate the actor
timeline = Timeline(playback_panel=Timeline)
# Adding the sphere actor to the timeline
# This could've been done during initialization.
timeline.add_actor(boxes)

# Adding some keyframes
timeline.set_position(0, np.array([0, 0, 0]))
timeline.set_position(2, np.array([10, 10, 10]))
timeline.set_position(5, np.array([-10, 0, 0]))
timeline.set_position(7, np.array([10, 0, 20]))

# change the position interpolator to Cubic spline interpolator.
timeline.set_position_interpolator(CubicSplineInterpolator)

# Main timeline to control all the timelines
scene.camera().SetPosition(0, 0, 60)

# Adding timelines to the main Timeline


# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


scene.add(timeline)

# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
