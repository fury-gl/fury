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
showm.initialize()

###############################################################################
# Cubic Bezier curve parameters
# =============================
# p0, p1, p2, 03
# p0 and p3 are the positions
# p1 is the control point for the point p0
# p2 is the control point for the point p3
points = [
    [-2, 0, 0],  # p0
    [-15, 6, 0],  # p1
    [27, 18, 0],  # p2
    [18, 0, 0],  # p3
]

###############################################################################
# Visualizing points
pts_actor = actor.sphere(np.array([points[0], points[3]]), (1, 0, 0),
                         radii=0.3)

###############################################################################
# Visualizing the control points
cps_actor = actor.sphere(np.array([points[1], points[2]]), (0, 0, 1),
                         radii=0.6)

###############################################################################
# Visualizing the connection between the control points and the points
cline_actor = actor.line(np.array([points[0:2], points[2:4]]),
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
# Timestamp, position, pre control point, and post control point.
# - Pre control point is the cubic bezier control point for the associated
#   position when this position is the destination position.
# - Post control point is the cubic bezier control point for the associated
#   position when this position is the origin position or departing position.
# Note: If a control point is not provided or set `None`, this control point
# will be the same as the position itself.

timeline.set_position(0, np.array(points[0]), out_cp=np.array(points[1]))
timeline.set_position(5, np.array(points[3]), in_cp=np.array(points[2]))
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
###############################################################################
# A more complex scene scene
# ==========================
#

scene = window.Scene()
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

###############################################################################
# Note: If a control point is set to `None`, it gets the value of the
# point it controls.
keyframes = [
    # time - position - pre control point  - post control point
    [0.0, [-2, 0, 0], None, [-15, 6, 0]],
    [5.0, [18, 0, 0], [27, 18, 0], [27, -18, 0]],
    [9.0, [-5, -10, -10], None, None]
]


###############################################################################
# Initializing the timeline
timeline = Timeline(playback_panel=True)
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))
timeline.add_actor(sphere)

###############################################################################
# Setting Cubic Bezier keyframes

for keyframe in keyframes:

    ###########################################################################
    # visualizing the points and control points (only for demonstration)
    t = keyframe[0]
    pos = keyframe[1]
    pre_control_point = keyframe[2]
    post_control_point = keyframe[3]

    ###########################################################################
    # setting position keyframe
    timeline.set_position(t, pos, in_cp=pre_control_point, out_cp=post_control_point)

    ###########################################################################
    # visualizing position keyframe
    vis_point = actor.sphere(np.array([pos]), (1, 0, 0), radii=0.3)
    scene.add(vis_point)

    ###########################################################################
    # Visualizing the control points and their length (if exist)
    for cp in [pre_control_point, post_control_point]:
        if cp is not None:
            vis_cps = actor.sphere(np.array([cp]), (0, 0, 1),
                                   radii=0.6)
            cline_actor = actor.line(np.array([[pos, cp]]),
                                     colors=np.array([0, 1, 0]))
            scene.add(vis_cps, cline_actor)

###############################################################################
# changing position interpolation into cubic bezier interpolation
timeline.set_position_interpolator(cubic_bezier_interpolator)

###############################################################################
# adding actors to the scene
scene.add(timeline)


###############################################################################
# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_bezier_2.png',
              size=(900, 768))
