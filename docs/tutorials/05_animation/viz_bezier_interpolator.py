"""
===================
Bezier Interpolator
===================

Keyframe animation using cubic Bezier interpolator.

"""
import numpy as np
from fury import actor, window
from fury.animation import Animation, Timeline
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
showm.initialize()

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
# Initializing an ``Animation`` and adding sphere actor to it.
animation = Animation()
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))
animation.add_actor(sphere)

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

animation.set_position(0.0, np.array(keyframe_1.get('value')),
                       out_cp=np.array(keyframe_1.get('out_cp')))
animation.set_position(5.0, np.array(keyframe_2.get('value')),
                       in_cp=np.array(keyframe_2.get('in_cp')))

###############################################################################
# Changing position interpolation into cubic bezier interpolation
animation.set_position_interpolator(cubic_bezier_interpolator)

###############################################################################
# Adding the visualization actors to the scene.
scene.add(pts_actor, cps_actor, cline_actor)

###############################################################################
# Adding the animation to the ``ShowManager``
showm.add_animation(animation)

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
show_manager.initialize()

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
# Creat the sphere actor.
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))

###############################################################################
# Creat an ``Animation`` and adding the sphere actor to it.
animation = Animation(sphere)

###############################################################################
# Setting Cubic Bezier keyframes
animation.set_position_keyframes(keyframes)

###############################################################################
# changing position interpolation into cubic bezier interpolation
animation.set_position_interpolator(cubic_bezier_interpolator)

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
# Initializing the timeline to be able to control the playback of the
# animation.
timeline = Timeline(animation, playback_panel=True)

###############################################################################
# We only need to add the ``Timeline`` to the ``ShowManager``
show_manager.add_animation(timeline)

###############################################################################
# Start the animation
if interactive:
    show_manager.start()

window.record(scene, out_path='viz_keyframe_animation_bezier_2.png',
              size=(900, 768))
