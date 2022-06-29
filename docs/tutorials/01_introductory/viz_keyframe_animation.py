
"""
=====================
Keyframe animation
=====================

Keyframe animation explained with a simple tutorial

"""

import numpy as np
from fury import actor, window, ui
from fury.animation import Timeline, StepInterpolator, LinearInterpolator, LABInterpolator, CubicSplineInterpolator
from fury.data import read_viz_icons

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

# creating the UI panel to hold the playback buttons
panel = ui.Panel2D(size=(250, 40), color=(1, 1, 1), align="right")
panel.center = (460, 40)


# creating 3 buttons to control the animation
pause_btn = ui.Button2D(
    icon_fnames=[("square", read_viz_icons(fname="pause2.png"))]
)
stop_btn = ui.Button2D(
    icon_fnames=[("square", read_viz_icons(fname="stop2.png"))]
)
start_btn = ui.Button2D(
    icon_fnames=[("square", read_viz_icons(fname="play3.png"))]
)

# Add the buttons on the panel
panel.add_element(pause_btn, (0.15, 0.15))
panel.add_element(start_btn, (0.45, 0.15))
panel.add_element(stop_btn, (0.75, 0.15))

# creating the actor to be animated
cube = actor.cube(np.array([[0, 0, 0]]), np.array([[1, 1, 1]]))

# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update()
    showm.render()


# simple adapters for the playback methods to be used on button left click

def start_animation(i_ren, _obj, _button):
    timeline.play()


def pause_animation(i_ren, _obj, _button):
    timeline.pause()


def stop_animation(i_ren, _obj, _button):
    timeline.stop()


# using the adapters created above
start_btn.on_left_mouse_button_clicked = start_animation
pause_btn.on_left_mouse_button_clicked = pause_animation
stop_btn.on_left_mouse_button_clicked = stop_animation

# adding actors to the scene
scene.add(cube)
scene.add(panel)

# Creating a timeline to animate the actor
timeline = Timeline([cube])

# Adding translation keyframes to the timeline at times 0, 5, and 12
timeline.translate(0, np.array([0, 0, 0]))
timeline.translate(3, np.array([-12, 11, 0]))
timeline.translate(6, np.array([12, 11, 5]))
timeline.translate(9, np.array([12, 11, 0]))

# Changing the default scale interpolator to be a step interpolator
# The default is linear interpolator for translation scale and color keyframes
# timeline.set_scale_interpolator(StepInterpolator)

# Adding scale keyframes to the timeline at times 0, 3, 6, 15
timeline.scale(0, np.array([1, 1,  1]))
timeline.scale(3, np.array([3, 3, 3]))
timeline.scale(6, np.array([2, 2, 2]))
timeline.scale(15, np.array([5, 5, 5]))

# Adding color keyframes to the timeline at times 0 and 7
timeline.set_color(0, np.array([1, 0, 0]))
timeline.set_color(7, np.array([0, 1, 0]))

# Adding multi property keyframes at the same timestamp.
timeline.set_keyframes(17, {
    "position": np.array([0, 0, 0]),
    "scale": np.array([1, 1, 1])
})


timeline.set_position_interpolator(CubicSplineInterpolator)


# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()

