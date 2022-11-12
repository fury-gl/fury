"""
=====================
Keyframe animation
=====================

Tutorial on making keyframe-based animation in FURY using Spline interpolators.

"""

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import spline_interpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


###############################################################################
# Position keyframes as a dict object containing timestamps as keys and
# positions as values.
position_keyframes = {
    0: np.array([0, 0, 0]),
    2: np.array([10, 3, 5]),
    4: np.array([20, 14, 13]),
    6: np.array([-20, 20, 0]),
    8: np.array([17, -10, 15]),
    10: np.array([0, -6, 0]),
}

###############################################################################
# creating FURY dots to visualize the position values.
pos_dots = actor.dot(np.array(list(position_keyframes.values())))

###############################################################################
# creating two timelines (one uses linear and the other uses' spline
# interpolator), each timeline controls a sphere actor
sphere_linear = actor.sphere(np.array([[0, 0, 0]]), (1, 0.5, 0.2), 0.5)
linear_tl = Timeline()
linear_tl.add(sphere_linear)

linear_tl.set_position_keyframes(position_keyframes)

###############################################################################
# Note: linear_interpolator is used by default. So, no need to set it for the
# first (linear position) timeline.

###############################################################################
# creating a second timeline that translates another larger sphere actor using
# spline interpolator.
sphere_spline = actor.sphere(np.array([[0, 0, 0]]), (0.3, 0.9, 0.6), 1)
spline_tl = Timeline(sphere_spline)
spline_tl.set_position_keyframes(position_keyframes)

###############################################################################
# Setting 5th degree spline interpolator for position keyframes.
spline_tl.set_position_interpolator(spline_interpolator, degree=5)

###############################################################################
# Adding everything to a main ``Timeline`` to control the two timelines.
# =============================================================================
#
###############################################################################
# Creating a timeline with a playback panel
main_timeline = Timeline(playback_panel=True, motion_path_res=100)

###############################################################################
# Add visualization dots actor to the timeline as a static actor.
main_timeline.add_static_actor(pos_dots)

###############################################################################
# Adding timelines to the main timeline (so that it controls their playback)
main_timeline.add([spline_tl, linear_tl])

###############################################################################
# Adding the main timeline to the scene.
scene.add(main_timeline)


###############################################################################
# Now that these two timelines are added to main_timeline, if main_timeline
# is played, paused, ..., all these changes will reflect on the children
# timelines.

###############################################################################
# making a function to update the animation and render the scene
def timer_callback(_obj, _event):
    main_timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_spline.png',
              size=(900, 768))
