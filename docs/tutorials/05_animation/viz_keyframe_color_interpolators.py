"""
==================
Keyframe animation
==================

Color animation explained

"""

import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import step_interpolator, \
    lab_color_interpolator, hsv_color_interpolator, xyz_color_interpolator

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


###############################################################################
# Initializing positions of the cubes that will be color-animated.
cubes_pos = np.array([
    [[-2, 0, 0]],
    [[0, 0, 0]],
    [[2, 0, 0]],
    [[4, 0, 0]],
    [[6, 0, 0]],
])

###############################################################################
# Static labels for different interpolators (for show)
linear_text = actor.vector_text("Linear", (-2.64, -1, 0))
lab_text = actor.vector_text("LAB", (-0.37, -1, 0))
hsv_text = actor.vector_text("HSV", (1.68, -1, 0))
xyz_text = actor.vector_text("XYZ", (3.6, -1, 0))
step_text = actor.vector_text("Step", (5.7, -1, 0))
scene.add(step_text, lab_text, linear_text, hsv_text, xyz_text)

###############################################################################
# Main timeline to control all the timelines (one for each color interpolation
# method)
main_timeline = Timeline(playback_panel=True)

###############################################################################
# Creating a timeline to animate the actor.
# Also cube actor is provided for each timeline to handle as follows:
# ``Timeline(actor)``, ``Timeline(list_of_actors)``, or actors can be added
# later using ``Timeline.add()`` or ``timeline.add_actor()``
timeline_linear_color = Timeline(actor.cube(cubes_pos[0]))
timeline_LAB_color = Timeline(actor.cube(cubes_pos[1]))
timeline_HSV_color = Timeline(actor.cube(cubes_pos[2]))
timeline_XYZ_color = Timeline(actor.cube(cubes_pos[3]))
timeline_step_color = Timeline(actor.cube(cubes_pos[4]))

###############################################################################
# Adding timelines to the main Timeline.
main_timeline.add_child_timeline([timeline_linear_color,
                                  timeline_LAB_color,
                                  timeline_HSV_color,
                                  timeline_XYZ_color,
                                  timeline_step_color])

###############################################################################
# Adding color keyframes to the linearly (for now) interpolated timelines
for t in range(0, 20, 5):
    x = np.random.random(3)
    timeline_linear_color.set_color(t, np.array(x))
    timeline_LAB_color.set_color(t, np.array(x))
    timeline_HSV_color.set_color(t, np.array(x))
    timeline_XYZ_color.set_color(t, np.array(x))
    timeline_step_color.set_color(t, np.array(x))

###############################################################################
# Changing the default scale interpolator to be a step interpolator
# The default is linear interpolator for color keyframes
timeline_HSV_color.set_color_interpolator(hsv_color_interpolator)
timeline_LAB_color.set_color_interpolator(lab_color_interpolator)
timeline_step_color.set_color_interpolator(step_interpolator)
timeline_XYZ_color.set_color_interpolator(xyz_color_interpolator)

###############################################################################
# Adding the main timeline to the scene
scene.add(main_timeline)


###############################################################################
# making a function to update the animation and render the scene
def timer_callback(_obj, _event):
    main_timeline.update_animation()
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_animation_colors.png',
              size=(900, 768))
