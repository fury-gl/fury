"""
======================================
Keyframe animation: Camera and opacity
======================================

Camera and opacity keyframe animation explained in this tutorial.
"""

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import cubic_spline_interpolator

###############################################################################
# The Plan
# ========
#
# The plan here is to animate (scale and translate) 50 spheres randomly, and
# show `FURY` text that appears at the end!

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


###############################################################################
# Creating the main ``Timeline`` and adding static actors to it
# =============================================================
#
# Here we create a ``Timeline``. which we will call ``main_timeline`` so that
# we can use it as a controller for the other 50 Timelines.
# So, Instead of updating and adding 50 timelines to the ``scene``, we only
# need to update the main ``Timeline``. Also, a playback panel can be assigned
# to this main Timeline.
# But, why we need 50 ``Timelines``, you may ask.
# -> A single ``Timeline`` can handle each property once at a time. So we need
# 50 ``Timelines`` to translate and scale our 50 spheres.

###############################################################################
# ``playback_panel=True`` assigns a playback panel that can control the
# playback of this ``main_timeline`` and all of its children ``Timelines``

main_timeline = Timeline(playback_panel=True)

###############################################################################
# Creating two actors for visualization, and to detect camera's animations.
arrow = actor.arrow(np.array([[0, 0, 0]]), np.array([[0, 1, 0]]),
                    np.array([[1, 1, 0]]), scales=5)
plan = actor.box(np.array([[0, 0, 0]]), colors=np.array([[1, 1, 1]]),
                 scales=np.array([[20, 0.2, 20]]))

###############################################################################
# adding static actors to the timeline.
# Note: adding actors as static actors just ensures that they get added to the
# scene along with the Timeline and will not be controlled nor animated by the
# timeline.
main_timeline.add_static_actor([arrow, plan])

###############################################################################
# Creating "FURY" text
# ====================
fury_text = actor.vector_text("FURY",
                              pos=(-4.3, 15, 0),
                              scale=(2, 2, 2))

###############################################################################
# Creating a ``Timeline`` to animate the opacity of ``fury_text``
text_timeline = Timeline(fury_text)

###############################################################################
# opacity is set to 0 at time 28 and set to one at time 31.
# Linear interpolator is always used by default.
text_timeline.set_opacity(29, 0)
text_timeline.set_opacity(35, 1)

###############################################################################
# ``text_timeline`` contains the text actor is added to the main Timeline.
main_timeline.add_child_timeline(text_timeline)

###############################################################################
# Creating and animating 50 Spheres
# =================================
#

for i in range(50):
    ###########################################################################
    # create a sphere actor that's centered at the origin and has random color
    # and radius.
    actors = [actor.sphere(np.array([[0, 0, 0]]),
                           np.random.random([1, 3]),
                           np.random.random([1, 3]))]

    ###########################################################################
    # create a timeline to animate this actor (single actor or list of actors)
    # Actors can be added later using `Timeline.add_actor(actor)`
    timeline = Timeline(actors)

    # We generate random position and scale values from time=0 to time=49 each
    # two seconds.
    for t in range(0, 50, 2):
        #######################################################################
        # Position and scale are set to a random value at the timestamps
        # mentioned above.
        timeline.set_position(t,
                              np.random.random(3) * 30 - np.array([15, 0, 15]))
        timeline.set_scale(t, np.repeat(np.random.random(1), 3))

    ###########################################################################
    # change the position interpolator to cubic spline interpolator.
    timeline.set_position_interpolator(cubic_spline_interpolator)

    ###########################################################################
    # Finally, the ``Timeline`` is added to the ``main_timeline``.
    main_timeline.add_child_timeline(timeline)

###############################################################################
# Animating the camera
# ====================
#
# Since, only one camera is needed, camera animations are preferably done using
# the main `Timeline`. Three properties can control the camera's animation:
# Position, focal position (referred to by `focal`), and up-view.

###############################################################################
# Multiple keyframes can be set at once as follows.

# camera focal positions
camera_positions = {
    # time: camera position
    0: np.array([3, 3, 3]),
    4: np.array([50, 25, -40]),
    7: np.array([-50, 50, -40]),
    10: np.array([-25, 25, 20]),
    14: np.array([0, 16, 25]),
    20: np.array([0, 14.5, 20]),
}

# camera focal positions
camera_focal_positions = {
    # time: focal position
    15: np.array([0, 0, 0]),
    20: np.array([3, 9, 5]),
    23: np.array([7, 5, 3]),
    25: np.array([-2, 9, -6]),
    27: np.array([0, 16, 0]),
    31: np.array([0, 14.5, 0]),
}

###############################################################################
# ``set_camera_focal`` can only set one keyframeB , but
# ``set_camera_focal_keyframes`` can set a dictionary of keyframes.
main_timeline.set_camera_focal_keyframes(camera_focal_positions)
main_timeline.set_camera_position_keyframes(camera_positions)

###############################################################################
# Change camera position and focal interpolators
main_timeline.set_camera_position_interpolator(cubic_spline_interpolator)
main_timeline.set_camera_focal_interpolator(cubic_spline_interpolator)

###############################################################################
# Only the main Timeline is added to the scene.
scene.add(main_timeline)


###############################################################################
# making a function to update the animation
def timer_callback(_obj, _event):
    ###########################################################################
    # Only the main timeline is needed to be updated, and it would update all
    # children ``Timelines``.
    main_timeline.update_animation()

    ###########################################################################
    # The scene is rendered after the animations are updated.
    showm.render()


###############################################################################
# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_camera.png',
              size=(900, 768))
