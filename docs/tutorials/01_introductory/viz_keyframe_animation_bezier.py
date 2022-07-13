"""
=====================
Keyframe animation
=====================

Keyframe animation using cubic Bezier interpolator.

"""
import numpy as np
from fury import actor, window
from fury.animation import Timeline, CubicBezierInterpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

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

# Visualizing points
pts_actor = actor.sphere(np.array([points[0], points[3]]), (1, 0, 0),
                         radii=0.3)
# Visualizing the control points
cps_actor = actor.sphere(np.array([points[1], points[2]]), (0, 0, 1),
                         radii=0.6)
# Visualizing the connection between the control points and the points
cline_actor = actor.line(np.array([points[0:2], points[2:4]]),
                         colors=np.array([0, 1, 0]))

timeline = Timeline(playback_panel=True)
sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1))
timeline.add_actor(sphere)

# Setting Cubic Bezier keyframes
timeline.set_position(0, np.array(points[0]), post_cp=np.array(points[1]))
timeline.set_position(5, np.array(points[3]), pre_cp=np.array(points[2]))
timeline.set_position_interpolator(CubicBezierInterpolator)

# adding actors to the scene
scene.add(pts_actor, cps_actor, cline_actor, timeline)
# getting the camera back a little


# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
