"""
=====================
Keyframe animation
=====================

Minimal tutorial of making keyframe-based animation in FURY.

"""

###############################################################################
# What is a ``Timeline``
# ======================
#
# ``Timeline`` is responsible for animating FURY actors using a set of
# keyframes by interpolating values between timestamps of these keyframes.
# ``Timeline`` has playback methods such as ``play``, ``pause``, ``stop``, ...

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import cubic_spline_interpolator, slerp

###############################################################################
# What are keyframes
# ==================
#
# A keyframe consists of a timestamp and some data.
# These data can be anything such as temperature, position, or scale.
# How to define Keyframes that FURY can work with?
# A simple dictionary object with timestamps as keys and data as values.

keyframes = {
    1.0: {'value': np.array([0, 0, 0])},
    2.0: {'value': np.array([-4, 1, 0])},
    5.0: {'value': np.array([0, 0, 12])},
    6.0: {'value': np.array([25, 0, 12])}
}

###############################################################################
# Why keyframes data are also a dictionary ``{'value': np.array([0, 0, 0])})``?
# -> Since some keyframes data can only be defined by a set of data i.e. a
# single position keyframe could consist of a position, in control point, and
# out control point or any other data that helps to define this keyframe.


###############################################################################
# What are the interpolators
# ==========================
#
# The keyframes interpolators are functions that takes a set of keyframes and
# returns a function that calculates an interpolated value between these
# keyframes.
# Below there is an example on how to use interpolators manually to interpolate
# the above defined ``keyframes``.

interpolation_function = cubic_spline_interpolator(keyframes)

###############################################################################
# Now, if we feed any time to this function it would return the cubic
# interpolated position at that time.

position = interpolation_function(1.44434)

###############################################################################
# ``position`` would contain an interpolated position at time equals 1.44434

###############################################################################
# Creating the environment
# ========================
#
# In order to make any animations in FURY, a `ShowManager` is needed to handle
# updating the animation and rendering the scene.

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

arrow = actor.arrow(np.array([[0, 0, 0]]), (0, 0, 0), (1, 0, 1), scales=6)

###############################################################################
# Creating the ``Timeline``
# ========================
#
# First step is creating the Timeline. A playback panel should be
timeline = Timeline(playback_panel=True)

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
timeline.add_actor(arrow)

###############################################################################
# Setting position keyframes
# ==========================
#
# Adding some position keyframes
timeline.set_position(0.0, np.array([0, 0, 0]))
timeline.set_position(2.0, np.array([10, 10, 10]))
timeline.set_position(5.0, np.array([-10, -3, -6]))
timeline.set_position(9.0, np.array([10, 6, 20]))

###############################################################################
# Changing the default interpolator for a single property
# =======================================================
#
# For all properties except **rotation**, linear interpolator is used by
# default. In order to change the default interpolator and set another
# interpolator, call ``timeline.set_<property>_interpolator(new_interpolator)``
# FURY already has some interpolators located at:
# ``fury.animation.interpolator``.
#
# Below we set the interpolator for position keyframes to be
# **cubic spline interpolator**.
timeline.set_position_interpolator(cubic_spline_interpolator)

###############################################################################
# Adding some rotation keyframes.
timeline.set_rotation(0.0, np.array([160, 50, 0]))
timeline.set_rotation(8.0, np.array([60, 160, 0]))

###############################################################################
# For Rotation keyframes, Slerp is used as the default interpolator.

###############################################################################
# Setting camera position to see the animation better.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding main Timeline to the ``Scene``.
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

window.record(scene, out_path='viz_keyframe_interpolator.png',
              size=(900, 768))
