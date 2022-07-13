"""
=====================
Keyframe animation
=====================

Color interpolation explained

"""

import numpy as np
from fury import actor, window
from fury.animation import Timeline, StepInterpolator, \
    LABInterpolator, HSVInterpolator, XYZInterpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

# creating the actors to be animated
cubes_pos = np.array([
    [[-2, 0, 0]],
    [[0, 0, 0]],
    [[2, 0, 0]],
    [[4, 0, 0]],
    [[6, 0, 0]],
])

# Static labels for different interpolators (for show)
linear_text = actor.vector_text("Linear", (-2.64, -1, 0))
lab_text = actor.vector_text("LAB", (-0.37, -1, 0))
hsv_text = actor.vector_text("HSV", (1.68, -1, 0))
xyz_text = actor.vector_text("XYZ", (3.6, -1, 0))
step_text = actor.vector_text("Step", (5.7, -1, 0))
scene.add(step_text, lab_text, linear_text, hsv_text, xyz_text)

# Creating a timeline to animate the actor
timeline_linear_color = Timeline(actor.cube(cubes_pos[0]))
timeline_LAB_color = Timeline(actor.cube(cubes_pos[1]))
timeline_HSV_color = Timeline(actor.cube(cubes_pos[2]))
timeline_XYZ_color = Timeline(actor.cube(cubes_pos[3]))
timeline_step_color = Timeline(actor.cube(cubes_pos[4]))

# Main timeline to control all the timelines
main_timeline = Timeline(playback_panel=Timeline)
main_timeline.set_camera_position(0, np.array([2, 0, 17]))
main_timeline.set_camera_focal(0, np.array([2, 0, 0]))

# Adding timelines to the main Timeline
main_timeline.add_timeline(timeline_linear_color)
main_timeline.add_timeline(timeline_LAB_color)
main_timeline.add_timeline(timeline_HSV_color)
main_timeline.add_timeline(timeline_step_color)
main_timeline.add_timeline(timeline_XYZ_color)

# Adding color keyframes to the linearly interpolated timeline
for t in range(0, 20, 5):
    x = np.random.random(3)
    timeline_linear_color.set_color(t, np.array(x))
    timeline_LAB_color.set_color(t, np.array(x))
    timeline_HSV_color.set_color(t, np.array(x))
    timeline_XYZ_color.set_color(t, np.array(x))
    timeline_step_color.set_color(t, np.array(x))

# Changing the default scale interpolator to be a step interpolator
# The default is linear interpolator for color keyframes
timeline_HSV_color.set_color_interpolator(HSVInterpolator)
timeline_LAB_color.set_color_interpolator(LABInterpolator)
timeline_step_color.set_color_interpolator(StepInterpolator)
timeline_XYZ_color.set_color_interpolator(XYZInterpolator)


# making a function to update the animation
def timer_callback(_obj, _event):
    main_timeline.update_animation()
    showm.render()


scene.add(main_timeline)

# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
