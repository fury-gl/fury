"""
===============================
Keyframe animation introduction
===============================

This tutorial explains keyframe animation in FURY.

"""

###############################################################################
# What is Keyframe Animation?
# ===========================
#
# A keyframe animation is the transition in a property setting in between two
# keyframes. That change could be anything.
#
# A Keyframe is simply a marker of time which stores the value of a property.
#
# For example, a Keyframe might define that the position of a FURY actor is
# at (0, 0, 0) at time equals 1 second.
#
# The purpose of a Keyframe is to allow for interpolated animation, meaning,
# for example, that the user could then add another key at time equals 3
# seconds, specifying the actor's position is at (1, 1, 0),
#
# Then the correct position of the actor for all the times between 3 and 10
# will be interpolated.
#
# Almost any parameter that you can set for FURY actors can be animated
# using keyframes.
#
# For this tutorial, we are going to use the FURY animation module to translate
# FURY sphere actor.

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline


scene = window.Scene()

showm = window.ShowManager(scene, size=(900, 768))
showm.initialize()


###############################################################################
# Translating a sphere
# ====================
#
# This is a quick demo showing how to translate a sphere from (0, 0, 0) to
# (1, 1, 1).
# First, we create a ``Timeline``
timeline = Timeline(playback_panel=True)

###############################################################################
# Our FURY sphere actor
sphere = actor.sphere(np.zeros([1, 3]), np.ones([1, 3]))

###############################################################################
# We add the sphere actor to the ``Timeline``
timeline.add_actor(sphere)

###############################################################################
# Then, we set our position keyframes at different timestamps
timeline.set_position(1, np.array([0, 0, 0]))
timeline.set_position(3, np.array([1, 1, 0]))

###############################################################################
# The ``Timeline`` must be added to the ``Scene``
scene.add(timeline)

###############################################################################
# No need to add the sphere actor, since it's now a part of the ``Timeline``


###############################################################################
# Now we have to update the animation using a timer callback function

def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


showm.add_timer_callback(True, 1, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_introduction.png',
              size=(900, 768))
