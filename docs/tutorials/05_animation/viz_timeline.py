"""
==============================
Timeline and setting keyframes
==============================

In his tutorial, you will learn how to use timeline to make simple keyframe
animations on FURY actors.

"""

###############################################################################
# What is ``Timeline``?
# ===================
# ``Timeline`` is responsible for handling keyframe animation of FURY actors
# using a set of keyframes by interpolating values between timestamps of these
# keyframes.
#
# ``Timeline`` has playback methods such as ``play``, ``pause``, ``stop``, ...
# which can be used to control the animation.
#

###############################################################################
# Creating a ``Timeline``
# =======================
#
# FURY ``Timeline`` has the option to attaches a very useful panel for
# controlling the animation by setting ``playback_panel=True``.

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline

timeline = Timeline(playback_panel=True)

###############################################################################
# Adding Actors to ``Timeline``
# ============================
#
# FURY actors must be added to the ``Timeline`` in order to be animated.

sphere = actor.sphere(np.zeros([1, 3]), np.ones([1, 3]))
timeline.add_actor(sphere)

###############################################################################
# Main use of ``Timeline`` is to handle playback (play, stop, ...) of the
# keyframe animation. But it can be used to animate FURY actors as well.
# Now that the actor is addd to the timeline, setting keyframes to the timeline
# will animate the actor accordingly.

###############################################################################
# Setting Keyframes
# =================
#
# There are multiple ways to set keyframes:
#
# 1- To set a single keyframe, you may use ``timeline.set_<property>(t, k)``,
# where <property> is the name of the property to be set. I.e. setting position
# to (1, 2, 3) at time 0.0 would be as following:
timeline.set_position(0.0, np.array([1, 2, 3]))

###############################################################################
# Supported properties are: **position, rotation, scale, color, and opacity**.
#
# 2- To set multiple keyframes at once, you may use
# ``timeline.set_<property>_keyframes(keyframes)``.
keyframes = {
    1.0: np.array([0, 0, 0]),
    3.0: np.array([-2, 0, 0])
}

timeline.set_position_keyframes(keyframes)

###############################################################################
# That's it! Now we are done setting keyframes.

scene = window.Scene()

showm = window.ShowManager(scene, size=(900, 768))
showm.initialize()

scene.add(timeline)


###############################################################################
# Now we have to update the ``Timeline`` animation then render the scene
# using a timer callback function.

def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


showm.add_timer_callback(True, 1, timer_callback)

interactive = True

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_timeline.png',
              size=(900, 768))
