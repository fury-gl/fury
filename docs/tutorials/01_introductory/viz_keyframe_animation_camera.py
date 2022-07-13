"""
======================================
Keyframe animation: Camera and opacity
======================================

Camera keyframe animation explained
in this tutorial

"""
import numpy as np
from fury import actor, window
from fury.animation import Timeline, CubicSplineInterpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

main_timeline = Timeline(playback_panel=True)

arrow = actor.arrow(np.array([[0, 0, 0]]), np.array([[0, 1, 0]]),
                    np.array([[1, 1, 0]]), scales=5)
plan = actor.box(np.array([[0, 0, 0]]), colors=np.array([[1, 1, 1]]),
                 scales=np.array([[20, 0.2, 20]]))

fury_text = actor.vector_text("FURY",
                              pos=(-4.3, 15, 0),
                              scale=(2, 2, 2))

# Text of 'FURY' that appears at the end.
# opacity is set to 0 at time 28 and set to one at time 31.
# Linear interpolator is always used by default.
text_timeline = Timeline(fury_text)
text_timeline.set_opacity(28, 0)
text_timeline.set_opacity(31, 1)

# The timeline contains the text actor is added to the main Timeline
# so that we only update this main timeline only without explicitly updating
# all the timelines
main_timeline.add_timeline(text_timeline)

for i in range(50):
    # create a sphere actor
    actors = [actor.sphere(np.array([[0, 0, 0]]),
                           np.random.random([1, 3]),
                           np.random.random([1, 3]))]
    # create a timeline to animate this actor (single actor or list of actors)
    # Actors can be added later using `Timeline.add_actor(actor)`
    timeline = Timeline(actors)

    for t in range(0, 50, 2):
        timeline.set_position(t,
                              np.random.random(3) * 30 - np.array([15, 0, 15]))
        timeline.set_scale(t, np.repeat(np.random.random(1), 3))

    # change the position interpolator to cubic spline interpolator
    timeline.set_position_interpolator(CubicSplineInterpolator)

    main_timeline.add_timeline(timeline)

# adding actors to the scene
scene.add(main_timeline, arrow, plan, fury_text)

# camera position animation
main_timeline.set_camera_position(0, np.array([3, 3, 3]))

# camera focal position animation
camera_focal_positions = {
    # time: focal position
    15: np.array([0, 0, 0]),
    20: np.array([3, 9, 5]),
    23: np.array([7, 5, 3]),
    25: np.array([-2, 9, -6]),
    27: np.array([0, 16, 0]),
    31: np.array([0, 14.5, 0]),
}
main_timeline.set_camera_focal_keyframes(camera_focal_positions)

# camera focal position animation
camera_positions = {
    # time: camera position
    0: np.array([3, 3, 3]),
    4: np.array([50, 25, -40]),
    7: np.array([-50, 50, -40]),
    10: np.array([-25, 25, 20]),
    14: np.array([0, 16, 25]),
    20: np.array([0, 14.5, 20]),
}
#
main_timeline.set_camera_position_keyframes(camera_positions)

# Change camera position and focal interpolators
main_timeline.set_camera_position_interpolator(CubicSplineInterpolator)
main_timeline.set_camera_focal_interpolator(CubicSplineInterpolator)


# making a function to update the animation
def timer_callback(_obj, _event):
    main_timeline.update_animation()
    showm.render()


# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
