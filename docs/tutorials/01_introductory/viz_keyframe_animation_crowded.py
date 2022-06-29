"""
=====================
Keyframe animation
=====================

Keyframe animation explained with a simple tutorial

"""
import random

import numpy as np
from fury import actor, window, ui
from fury.animation import Timeline, LinearInterpolator, LABInterpolator, CubicSplineInterpolator
from fury.data import read_viz_icons

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

# creating the UI panel to hold the playback buttons
panel = ui.Panel2D(size=(150, 30), color=(1, 1, 1), align="right",
                   has_border=True, border_color=(0, 0.3, 0), border_width=2)
panel.center = (160 / 2, 40 / 2)

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
panel.add_element(pause_btn, (0.15, 0.04))
panel.add_element(start_btn, (0.45, 0.04))
panel.add_element(stop_btn, (0.7, 0.04))

# creating the actor to be animated
cube = actor.cube(np.array([[0, 0, 0]]), np.array([[1, 1, 1]]))

# creating walls
walls = actor.square(np.array([[-7, 0, -4], [7, 0, -4], [0, 0, -4]]),
                     np.array([[-1, 0, 1], [-1, 0, -1], [0, 0, 0]]),
                     colors=np.array([[1, 1, 1]]), scales=50)


# simple adapters for the playback methods to be used on button left click

def start_animation(i_ren, _obj, _button):
    for timeline in timelines:
        timeline.play()


def pause_animation(i_ren, _obj, _button):
    for timeline in timelines:
        timeline.pause()


def stop_animation(i_ren, _obj, _button):
    for timeline in timelines:
        timeline.stop()


# using the adapters created above
start_btn.on_left_mouse_button_clicked = start_animation
pause_btn.on_left_mouse_button_clicked = pause_animation
stop_btn.on_left_mouse_button_clicked = stop_animation

# adding actors to the scene
scene.add(cube)
scene.add(walls)
scene.add(panel)

timelines = []

for i in range(100):
    actors = [
        actor.cube(np.array([[0, 0, 0]]), np.random.random([1, 3]), np.random.random([1, 3])),
        actor.octagonalprism(np.array([[0, 0, 0]]), np.random.random([1, 3]), np.random.random([1, 3])),
        actor.sphere(np.array([[0, 0, 0]]), np.random.random([1, 3])),
    ]
    act = random.choice(actors)
    t = Timeline(act)
    scene.add(act)

    c = 0
    for time in range(0, 100, 1):
        if random.random() < 0.1:
            t.translate(time, np.random.random(3) * 600 - 300)

        if random.random() < 0.2:
            s = random.random() * 27
            t.scale(time, np.array([s] * 3))

        t.set_color(time, np.random.random(3))

    timelines.append(t)

for t in timelines:
    # Cubic interpolator has some rules, and it might not be satisfied all the times.
    try:
        t.set_position_interpolator(CubicSplineInterpolator)
    except:
        t.set_position_interpolator(LinearInterpolator)

    t.set_color_interpolator(LABInterpolator)

max_timestamp = max(i.last_timestamp for i in timelines)
timeline_slider = ui.LineSlider2D(center=(450 + 150 / 2, 20), initial_value=0,
                                  orientation='horizontal',
                                  min_value=0, max_value=100,
                                  text_alignment='bottom', length=700, line_width=9)


def change_timestamp(slider):
    global max_timestamp, timelines
    new_timestamp = slider.value * max_timestamp / 100.0
    [tl.set_timestamp(new_timestamp) for tl in timelines]


timeline_slider.on_change = change_timestamp

scene.add(timeline_slider)


# making a function to update the animation
def timer_callback(_obj, _event):
    for timeline in timelines:
        timeline.update()
    timeline_slider.value = timelines[0].current_timestamp * 100 / max_timestamp
    showm.render()


# Adding the callback function that updates the animation
showm.add_timer_callback(True, 1, timer_callback)

showm.start()
