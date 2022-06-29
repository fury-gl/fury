"""
=====================
Keyframe animation
=====================

Keyframe animation explained with a simple tutorial

"""

import numpy as np
from fury import actor, window, ui
from fury.animation import Timeline, StepInterpolator, LinearInterpolator, LABInterpolator, HSVInterpolator
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

# creating the actors to be animated
cube = actor.cube(np.array([[-2, 0, 0]]))
cube1 = actor.cube(np.array([[0, 0, 0]]))
cube2 = actor.cube(np.array([[2, 0, 0]]))
cube3 = actor.cube(np.array([[4, 0, 0]]))


# making a function to update the animation
def timer_callback(_obj, _event):
    timeline_linear_color.update()
    timeline_LAB_color.update()
    timeline_HSV_color.update()
    timeline_step_color.update()
    showm.render()


# Creating a timeline to animate the actor
timeline_linear_color = Timeline(cube)
timeline_LAB_color = Timeline(cube1)
timeline_HSV_color = Timeline(cube2)
timeline_step_color = Timeline(cube3)


# simple adapters for the playback methods to be used on button left click

def start_animation(i_ren, _obj, _button):
    timeline_linear_color.play()
    timeline_LAB_color.play()
    timeline_HSV_color.play()
    timeline_step_color.play()


def pause_animation(i_ren, _obj, _button):
    timeline_linear_color.pause()
    timeline_LAB_color.pause()
    timeline_HSV_color.pause()
    timeline_step_color.pause()


def stop_animation(i_ren, _obj, _button):
    timeline_linear_color.stop()
    timeline_LAB_color.stop()
    timeline_HSV_color.stop()
    timeline_step_color.stop()


# using the adapters created above
start_btn.on_left_mouse_button_clicked = start_animation
pause_btn.on_left_mouse_button_clicked = pause_animation
stop_btn.on_left_mouse_button_clicked = stop_animation

# labels
lab_text = actor.vector_text("Linear", (-2.64, -1, 0))
linear_text = actor.vector_text("LAB", (-0.37, -1, 0))
hsv_text = actor.vector_text("HSV", (1.68, -1, 0))
step_text = actor.vector_text("Step", (3.36, -1, 0))

# adding actors to the scene
scene.add(cube)
scene.add(cube1)
scene.add(cube2)
scene.add(cube3)

scene.add(panel)
scene.add(lab_text)
scene.add(linear_text)
scene.add(hsv_text)
scene.add(step_text)

k_frames = [
    (0, [1, 0, 0]),
    (3, [0, 1, 0]),
    (10, [0.24, 0.1, 0.6])]

for k in k_frames:
    # Adding color keyframes to the linearly interpolated timeline
    timeline_linear_color.set_color(k[0], np.array(k[1]))

    # Adding color keyframes to the LAB interpolator
    timeline_LAB_color.set_color(k[0], np.array(k[1]))

    # Adding color keyframes to the LAB interpolator
    timeline_HSV_color.set_color(k[0], np.array(k[1]))

    # Adding color keyframes to the LAB interpolator
    timeline_step_color.set_color(k[0], np.array(k[1]))


# Changing the default scale interpolator to be a step interpolator
# The default is linear interpolator for color keyframes
timeline_HSV_color.set_color_interpolator(HSVInterpolator)
timeline_LAB_color.set_color_interpolator(LABInterpolator)
timeline_step_color.set_color_interpolator(StepInterpolator)

# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
