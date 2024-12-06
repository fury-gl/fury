"""
=====================
Keyframe animation
=====================

Minimal tutorial of making keyframe-based animation in FURY.
"""

###############################################################################
# What is an ``Animation``
# ========================
#
# ``Animation`` is responsible for animating FURY actors using a set of
# keyframes by interpolating values between timestamps of these keyframes.

import numpy as np

import fury

keyframes = {
    1.0: {"value": np.array([0, 0, 0])},
    2.0: {"value": np.array([-4, 1, 0])},
    5.0: {"value": np.array([0, 0, 12])},
    6.0: {"value": np.array([25, 0, 12])},
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

interpolation_function = fury.animation.cubic_spline_interpolator(keyframes)

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

scene = fury.window.Scene()

showm = fury.window.ShowManager(
    scene=scene, size=(900, 768), reset_camera=False, order_transparent=True
)
showm.initialize()

arrow = fury.actor.arrow(np.array([[0, 0, 0]]), (0, 0, 0), (1, 0, 1), scales=6)

###############################################################################
# Creating an ``Animation``
# =========================
#
# First step is creating the Animation.
animation = fury.animation.Animation()

###############################################################################
# Adding the sphere actor to the timeline
# This could've been done during initialization.
animation.add_actor(actor=arrow)

###############################################################################
# Setting position keyframes
# ==========================
#
# Adding some position keyframes
animation.set_position(0.0, np.array([0, 0, 0]))
animation.set_position(2.0, np.array([10, 10, 10]))
animation.set_position(5.0, np.array([-10, -3, -6]))
animation.set_position(9.0, np.array([10, 6, 20]))

###############################################################################
# Changing the default interpolator for a single property
# =======================================================
#
# For all properties except **rotation**, linear interpolator is used by
# default. In order to change the default interpolator and set another
# interpolator, call ``animation.set_<property>_interpolator(interpolator)``
# FURY already has some interpolators located at:
# ``fury.animation.interpolator``.
#
# Below we set the interpolator for position keyframes to be
# **cubic spline interpolator**.
animation.set_position_interpolator(fury.animation.cubic_spline_interpolator)

###############################################################################
# Adding some rotation keyframes.
animation.set_rotation(0.0, np.array([160, 50, 0]))
animation.set_rotation(8.0, np.array([60, 160, 0]))

###############################################################################
# For Rotation keyframes, Slerp is used as the default interpolator.
# What is Slerp?
# Slerp (spherical linear interpolation) of quaternions results in a constant
# speed rotation in keyframe animation.
# Reed more about Slerp: https://en.wikipedia.org/wiki/Slerp

###############################################################################
# Setting camera position to see the animation better.
scene.set_camera(position=(0, 0, 90))

###############################################################################
# Adding main animation to the ``ShowManager``.
showm.add_animation(animation)

###############################################################################
# Start the ``ShowManager`` to start playing the animation
interactive = False

if interactive:
    showm.start()

fury.window.record(
    scene=scene, out_path="viz_keyframe_interpolator.png", size=(900, 768)
)
