"""
========
Tab UI
========

This example shows how to use the Tab UI. We will demonstrate how to
create Tabs for:

1. Slider controls for a Cube
2. Checkboxes for Cylinder and Sphere
3. Color combobox for Fury.

First, some imports.
"""
from fury import ui, window, actor
import numpy as np

###############################################################################
# First, we create the Tab UI.

tab_ui = ui.TabUI(position=(49, 94), size=(300, 300), nb_tabs=3,
                  draggable=True)

###############################################################################
# Slider Controls for a Cube
# ==========================
#
# Now we prepare content for the first tab.

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

cube = actor.box(centers=np.array([[10, 0, 0]]),
                 directions=np.array([[0, 1, 0]]),
                 colors=np.array([[0, 0, 255]]),
                 scale=np.array([[1, 1, 1]]))
cube_x = 0
cube_y = 0

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

ring_slider.on_change = rotate_cube
line_slider_x.on_change = translate_cube_x
line_slider_y.on_change = translate_cube_y

###############################################################################
# After defining content, we define properties for the tab.

tab_ui.tabs[0].title = "Sliders"
tab_ui.tabs[0].add_element(ring_slider, (0.3, 0.3))
tab_ui.tabs[0].add_element(line_slider_x, (0.0, 0.0))
tab_ui.tabs[0].add_element(line_slider_y, (0.0, 0.1))

###############################################################################
# CheckBoxes For Cylinder and Sphere
# ==============================
#
# Now we prepare content for second tab.

cylinder = actor.cylinder(centers=np.array([[0, 0, 0]]),
                          directions=np.array([[0, 1, 0]]),
                          colors=np.array([[0, 1, 1]]))

sphere = actor.sphere(centers=np.array([[5, 0, 0]]),
                      colors=(1, 1, 0))

figure_dict = {'cylinder': cylinder, 'sphere': sphere}
checkbox = ui.Checkbox(labels=["cylinder", "sphere"])

# Get difference between two lists.
def sym_diff(l1, l2):
    return list(set(l1).symmetric_difference(set(l2)))

# Set Visiblity of the figures
def set_figure_visiblity(checkboxes):
    checked = checkboxes.checked_labels
    unchecked = sym_diff(list(figure_dict), checked)

    for visible in checked:
        figure_dict[visible].SetVisibility(True)

    for invisible in unchecked:
        figure_dict[invisible].SetVisibility(False)

checkbox.on_change = set_figure_visiblity

###############################################################################
# After defining content, we define properties for the tab.

tab_ui.tabs[1].title = "Checkbox"
tab_ui.tabs[1].add_element(checkbox, (0.2, 0.2))

###############################################################################
# Define on_change & on_collapsed methods for tab ui to perform certain tasks
# while active tab is changed or when the tab is collapsed.

def hide_actors(tab_ui):
    if tab_ui.tabs[tab_ui.active_tab_idx].title == "Sliders":
        cube.SetVisibility(True)
        cylinder.SetVisibility(False)
        sphere.SetVisibility(False)

    elif tab_ui.tabs[tab_ui.active_tab_idx].title == "Checkbox":
        cube.SetVisibility(False)
        set_figure_visiblity(checkbox)

    else:
        pass

def collapse(tab_ui):
    if tab_ui.collapsed:
        cube.SetVisibility(False)
        cylinder.SetVisibility(False)
        sphere.SetVisibility(False)

tab_ui.on_change = hide_actors
tab_ui.on_collapse = collapse


###############################################################################
# Next we prepare the scene and render it with the help of show manager.

sm = window.ShowManager(size=(800, 500), title="Viz Tab")
sm.scene.add(tab_ui, cube, cylinder, sphere)

# To interact with the ui set interactive = True
interactive = True

if interactive:
    sm.start()

window.record(sm.scene, size=(500, 500), out_path="viz_tab.png")
