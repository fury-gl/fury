# -*- coding: utf-8 -*-
"""
===============
Buttons & Text
===============

This example shows how to use the UI API. We will demonstrate how to create
panel having buttons with callbacks.

First, some imports.
"""
from fury import ui, window
from fury.data import read_viz_icons

###############################################################################
# Let's create some buttons and text and put them in a panel.
# First we'll make the panel.

panel = ui.Panel2D(size=(300, 150), color=(1, 1, 1), align="right")
panel.center = (500, 400)

###############################################################################
# Then we'll make two text labels and place them on the panel.
# Note that we specifiy the position with integer numbers of pixels.

text = ui.TextBlock2D(text="Click me")
text2 = ui.TextBlock2D(text="Me too")
panel.add_element(text, (50, 100))
panel.add_element(text2, (180, 100))

###############################################################################
# Then we'll create two buttons and add them to the panel.
#
# Note that here we specify the positions with floats. In this case, these are
# percentages of the panel size.


button_example = ui.Button2D(
    icon_fnames=[("square", read_viz_icons(fname="stop2.png"))]
)

icon_files = []
icon_files.append(("down", read_viz_icons(fname="circle-down.png")))
icon_files.append(("left", read_viz_icons(fname="circle-left.png")))
icon_files.append(("up", read_viz_icons(fname="circle-up.png")))
icon_files.append(("right", read_viz_icons(fname="circle-right.png")))

second_button_example = ui.Button2D(icon_fnames=icon_files)

panel.add_element(button_example, (0.25, 0.33))
panel.add_element(second_button_example, (0.66, 0.33))

###############################################################################
# We can add a callback to each button to perform some action.


def change_text_callback(i_ren, _obj, _button):
    text.message = "Clicked!"
    i_ren.force_render()


def change_icon_callback(i_ren, _obj, _button):
    _button.next_icon()
    i_ren.force_render()


button_example.on_left_mouse_button_clicked = change_text_callback
second_button_example.on_left_mouse_button_pressed = change_icon_callback

###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="DIPY Button Example")

show_manager.scene.add(panel)

interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size,
              out_path="viz_button.png")
