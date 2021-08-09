# -*- coding: utf-8 -*-
"""
==============================
Actor Modifier using a Tree UI
==============================

This example shows how to create an actor moidifier using a Tree UI.
The parameters that will be modified are the colors, position,
rotation of the cube.

First, some imports.
"""
import numpy as np
from fury import ui, window, actor, utils

structure = [{'Cube': ['Translate', 'Color']},
             {'Cylinder': []},
             {'Cone': []}]

tree = ui.elements.Tree2D(structure=structure, tree_name="Actor Modifier",
                          size=(400, 400), position=(0, 400),
                          color=(0.8, 0.4, 0.2), opacity=1)

###############################################################################
# Slider Controls for the node Cube
# ==========================================
#
# Now we prepare content for the Cube node.

ring_slider = ui.RingSlider2D(initial_value=0,
                              text_template="{angle:5.1f}°")

line_slider_x = ui.LineSlider2D(initial_value=0,
                                min_value=-10, max_value=10,
                                orientation="horizontal",
                                text_alignment="Top")

line_slider_y = ui.LineSlider2D(initial_value=0,
                                min_value=-10, max_value=10,
                                orientation="vertical",
                                text_alignment="Right")

line_slider_r = ui.LineSlider2D(initial_value=0,
                                min_value=0, max_value=1,
                                orientation="vertical",
                                text_alignment="Left")

line_slider_g = ui.LineSlider2D(initial_value=0,
                                min_value=0, max_value=1,
                                orientation="vertical",
                                text_alignment="Left")

line_slider_b = ui.LineSlider2D(initial_value=0,
                                min_value=0, max_value=1,
                                orientation="vertical",
                                text_alignment="Left")

cube = actor.box(centers=np.array([[10, 0, 0]]),
                 directions=np.array([[0, 1, 0]]),
                 colors=np.array([[0, 0, 1]]),
                 scales=np.array([[0.3, 0.3, 0.3]]))

cube_x = 0
cube_y = 0
cube_r = 0
cube_g = 0
cube_b = 0


def rotate_cube(slider):
    angle = slider.value
    previous_angle = slider.previous_value
    rotation_angle = angle - previous_angle
    cube.RotateX(rotation_angle)


def translate_cube_x(slider):
    global cube_x, cube_y
    cube_x = slider.value
    cube.SetPosition(cube_x, cube_y, 0)


def translate_cube_y(slider):
    global cube_x, cube_y
    cube_y = slider.value
    cube.SetPosition(cube_x, cube_y, 0)


def update_colors():
    global cube
    vcolors = utils.colors_from_actor(cube)
    colarr = np.array([cube_r, cube_g, cube_b])*255
    vcolors[:] = colarr
    utils.update_actor(cube)


def change_r(slider):
    global cube_r, cube_g, cube_b
    cube_r = slider.value
    update_colors()


def change_g(slider):
    global cube_r, cube_g, cube_b
    cube_g = slider.value
    update_colors()


def change_b(slider):
    global cube_r, cube_g, cube_b
    cube_b = slider.value
    update_colors()


ring_slider.on_change = rotate_cube
line_slider_x.on_change = translate_cube_x
line_slider_y.on_change = translate_cube_y


#  Callbacks for color sliders
line_slider_r.on_change = change_r
line_slider_g.on_change = change_g
line_slider_b.on_change = change_b

###############################################################################
# Adding sliders to their respective nodes

tree.add_content('Translate', ring_slider, (0.5, 0.5))
tree.add_content('Translate', line_slider_x, (0, 0.))
tree.add_content('Translate', line_slider_y, (0., 0.))

tree.add_content('Color', line_slider_r, (0., 0.))
tree.add_content('Color', line_slider_g, (0.25, 0.))
tree.add_content('Color', line_slider_b, (0.5, 0.))

###############################################################################
# Defining hook to toggle the visibility of the cube


def visibility_on(tree_ui):
    global cube
    cube.SetVisibility(1)


def visibility_off(tree_ui):
    global cube
    cube.SetVisibility(0)

###############################################################################
# Adding hooks to relevant nodes

cube_node = tree.select_node('Cube')
color_node = tree.select_node('Color')

cube_node.on_node_select = visibility_on
cube_node.on_node_deselect = visibility_off

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Tree2D Example")

show_manager.scene.add(tree, cube)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size,
              out_path="viz_tree_actor_modifier.png")
