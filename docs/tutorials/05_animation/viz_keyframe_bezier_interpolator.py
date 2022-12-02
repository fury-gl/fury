"""
=====================
Keyframe animation
=====================

Keyframe animation using cubic Bezier interpolator.

"""
import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import cubic_bezier_interpolator

###############################################################################
# Position interpolation using cubic Bezier curve
# ===============================================
#
# Cubic bezier curve is a widely used method for interpolating motion paths.
# This can be achieved using positions and control points between those
# positions.

scene = window.Scene()
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


###############################################################################
# Cubic Bezier curve parameters
# =============================
# In order to make a cubic bezier curve based animation, you need four values
# for every keyframe:
# 1- Timestamp: The time that the keyframe is assigned to.
# 2- value: The value of the keyframe. This might be position, quaternion, or
#           scale value.
# 3- In control point: The control point used when the value is the destination
#           value.
# 4- Out control point: The control point used when the value is the departure
#           value.
#
#         keyframe 0            ----------------->         keyframe 1
# (time-0) (value-0) (out-cp-0) -----------------> (time-1) (value-1) (in-cp-1)
#
#         keyframe 1            ----------------->         keyframe 2
# (time-1) (value-1) (out-cp-1) -----------------> (time-2) (value-2) (in-cp-2)

keyframe_1 = {'value': [-2, 0, 0], 'out_cp': [-15, 6, 0]}
keyframe_2 = {'value': [18, 0, 0], 'in_cp': [27, 18, 0]}

###############################################################################
# Visualizing points
pts_actor = actor.sphere(np.array(
    [keyframe_1.get('value'), keyframe_2.get('value')]), (1, 0, 0), radii=0.3)

###############################################################################
# Visualizing the control points
cps_actor = actor.sphere(np.array(
    [keyframe_2.get('in_cp'), keyframe_1.get('out_cp')]), (0, 0, 1), radii=0.6)

###############################################################################
# Visualizing the connection between the control points and the points
cline_actor = actor.line(np.array([list(keyframe_1.values()),
                                   list(keyframe_2.values())]),
                         colors=np.array([0, 1, 0]))

###############################################################################
# Initializing a ``Timeline`` and adding sphere actor to it.
timeline = Timeline(playback_panel=True)
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))
timeline.add_actor(sphere)

###############################################################################
# Setting Cubic Bezier keyframes
# ==============================
#
# Cubic Bezier keyframes consists of 4 data per keyframe
# Timestamp, position, in control point, and out control point.
# - In control point is the cubic bezier control point for the associated
#   position when this position is the destination position.
# - Out control point is the cubic bezier control point for the associated
#   position when this position is the origin position or departing position.
# Note: If a control point is not provided or set `None`, this control point
# will be the same as the position itself.

timeline.set_position(0, np.array(keyframe_1.get('value')),
                      out_cp=np.array(keyframe_1.get('out_cp')))
timeline.set_position(5, np.array(keyframe_2.get('value')),
                      in_cp=np.array(keyframe_2.get('in_cp')))

###############################################################################
# changing position interpolation into cubic bezier interpolation
timeline.set_position_interpolator(cubic_bezier_interpolator)

###############################################################################
# adding the timeline and the static actors to the scene.
scene.add(pts_actor, cps_actor, cline_actor)
scene.add(timeline)


###############################################################################
# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_bezier_1.png',
              size=(900, 768))

###############################################################################
# A more complex scene scene
# ==========================
#

scene = window.Scene()
show_manager = window.ShowManager(scene,
                                  size=(900, 768), reset_camera=False,
                                  order_transparent=True)

###############################################################################
# Note: If a control point is set to `None`, it gets the value of the
# point it controls.
keyframes = {
    # time - position - in control point  - out control point
    0.0: {'value': [-2, 0, 0], 'out_cp': [-15, 6, 0]},
    5.0: {'value': [18, 0, 0], 'in_cp': [27, 18, 0], 'out_cp': [27, -18, 0]},
    9.0: {'value': [-5, -10, -10]}
}

###############################################################################
# Initializing the timeline
timeline = Timeline(playback_panel=True)
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))
timeline.add_actor(sphere)

###############################################################################
# Setting Cubic Bezier keyframes
timeline.set_position_keyframes(keyframes)

###############################################################################
# changing position interpolation into cubic bezier interpolation
timeline.set_position_interpolator(cubic_bezier_interpolator)

###########################################################################
# visualizing the points and control points (only for demonstration)
for t, keyframe in keyframes.items():
    pos = keyframe.get('value')
    in_control_point = keyframe.get('in_cp')
    out_control_point = keyframe.get('out_cp')

    ###########################################################################
    # visualizing position keyframe
    vis_point = actor.sphere(np.array([pos]), (1, 0, 0), radii=0.3)
    scene.add(vis_point)

    ###########################################################################
    # Visualizing the control points and their length (if exist)
    for cp in [in_control_point, out_control_point]:
        if cp is not None:
            vis_cps = actor.sphere(np.array([cp]), (0, 0, 1),
                                   radii=0.6)
            cline_actor = actor.line(np.array([[pos, cp]]),
                                     colors=np.array([0, 1, 0]))
            scene.add(vis_cps, cline_actor)

###############################################################################
# adding actors to the scene
scene.add(timeline)


###############################################################################
# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    show_manager.render()


###############################################################################
# Adding the callback function that updates the animation
show_manager.add_timer_callback(True, 10, timer_callback)

if interactive:
    show_manager.start()

window.record(scene, out_path='viz_keyframe_animation_bezier_2.png',
              size=(900, 768))
