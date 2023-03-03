"""
==============================
Timeline and setting keyframes
==============================

In his tutorial, you will learn how to use Fury ``Timeline`` for playing the
animations.
"""

###############################################################################
# What is ``Timeline``?
# =====================
#
# ``Timeline`` is responsible for handling the playback of Fury Animations.
#
# ``Timeline`` has playback methods such as ``play``, ``pause``, ``stop``, ...
# which can be used to control the animation.


import numpy as np

from fury import actor, window
from fury.animation import Animation, Timeline

###############################################################################
# We create our ``Scene`` and ``ShowManager`` as usual.
scene = window.Scene()

showm = window.ShowManager(scene, size=(900, 768))

###############################################################################
# Creating a ``Timeline``
# =======================
#
# FURY ``Timeline`` has the option to attaches a very useful panel for
# controlling the animation by setting ``playback_panel=True``.

###############################################################################
# Creating a ``Timeline`` with a PlaybackPanel.
timeline = Timeline(playback_panel=True)

###############################################################################
# Creating a Fury Animation as usual
anim = Animation()
sphere = actor.sphere(np.zeros([1, 3]), np.ones([1, 3]))
anim.add_actor(sphere)
# Now that the actor is addd to the ``Animation``, setting keyframes to the
# Animation will animate the actor accordingly.


###############################################################################
# Setting Keyframes
# =================
#
# There are multiple ways to set keyframes:
#
# 1- To set a single keyframe, you may use ``animation.set_<property>(t, k)``,
# where <property> is the name of the property to be set. I.e. setting position
# to (1, 2, 3) at time 0.0 would be as following:
anim.set_position(0.0, np.array([1, 2, 3]))

###############################################################################
# Supported properties are: **position, rotation, scale, color, and opacity**.
#
# 2- To set multiple keyframes at once, you may use
# ``animation.set_<property>_keyframes(keyframes)``.
keyframes = {1.0: np.array([0, 0, 0]), 3.0: np.array([-2, 0, 0])}

anim.set_position_keyframes(keyframes)

###############################################################################
# That's it! Now we are done setting keyframes.

###############################################################################
# In order to control this animation by the timeline we created earlier, this
# animation must be added to the timeline.
timeline.add_animation(anim)

###############################################################################
# Now we add only the ``Timeline`` to the ``ShowManager`` the same way we add
# ``Animation`` to the ``ShowManager``.
showm.add_animation(timeline)

scene.set_camera(position=(0, 0, -10))

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_timeline.png', size=(900, 768))
