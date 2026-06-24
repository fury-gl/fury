"""
==============================
Actor Modifier using a Tree UI
==============================

This example shows how to create an actor modifier using a Tree UI.
The parameters that will be modified are the colors, position, and
rotation of the actors.
"""

###############################################################################
# Adding the imports

import numpy as np
from fury import ui, window, actor

###############################################################################
# Defining the UI Structure

structure = [
    {"Cube": ["Translate", "Color"]},
    {"Cylinder": ["Translate", "Color"]},
    {"Cone": ["Translate", "Color"]},
]

###############################################################################
# Next, we instantiate the ``Tree2D`` object using the structure defined above.

tree = ui.elements.Tree2D(
    structure=structure,
    tree_name="Actor Modifier",
    size=(500, 500),
    position=(0, 0),
    color=(0.8, 0.4, 0.2),
    opacity=1,
    multiselect=False,
)

###############################################################################
# Creating the Actors

cube = actor.box(
    centers=np.array([[10, 0, 0]]),
    directions=np.array([[0, 1, 0]]),
    colors=np.array([[0, 0, 1]]),
    scales=np.array([[0.3, 0.3, 0.3]]),
)

cylinder = actor.cylinder(
    centers=np.array([[10, 0, 0]]),
    directions=np.array([[0, 1, 0]]),
    colors=np.array([[0, 1, 0]]),
    radii=0.2,
    height=0.4,
)

cone = actor.cone(
    centers=np.array([[10, 0, 0]]),
    directions=np.array([[0, 1, 0]]),
    colors=np.array([[1, 0, 0]]),
    radii=0.2,
    height=0.4,
)

cube.visible = False
cylinder.visible = False
cone.visible = False

###############################################################################
# Building the Slider Controls


def setup_actor_controls(actor_obj, node_name):
    # 1. Create the UI Sliders
    ring_slider = ui.RingSlider2D(initial_value=0, text_template="{angle:5.1f}°")

    line_slider_x = ui.LineSlider2D(
        initial_value=0, min_value=-10, max_value=10, orientation="horizontal"
    )
    line_slider_y = ui.LineSlider2D(
        initial_value=0, min_value=-10, max_value=10, orientation="vertical"
    )
    line_slider_r = ui.LineSlider2D(
        initial_value=0, min_value=0, max_value=1, orientation="vertical"
    )
    line_slider_g = ui.LineSlider2D(
        initial_value=0, min_value=0, max_value=1, orientation="vertical"
    )
    line_slider_b = ui.LineSlider2D(
        initial_value=0, min_value=0, max_value=1, orientation="vertical"
    )

    # To store the current state for this actor
    class State:
        x = 0.0
        y = 0.0
        r = 0.0
        g = 0.0
        b = 0.0

    state = State()

    # 2. Define the Callbacks
    def rotate(slider):
        actor_obj.local.euler_x = np.radians(slider.value)

    def translate_x(slider):
        state.x = slider.value
        actor_obj.local.position = (state.x, state.y, 0)

    def translate_y(slider):
        state.y = slider.value
        actor_obj.local.position = (state.x, state.y, 0)

    def update_colors():
        actor_obj.material.color_mode = "uniform"
        actor_obj.material.color = (state.r, state.g, state.b, 1.0)

    def change_r(slider):
        state.r = slider.value
        update_colors()

    def change_g(slider):
        state.g = slider.value
        update_colors()

    def change_b(slider):
        state.b = slider.value
        update_colors()

    # Attach the callbacks to the sliders
    ring_slider.on_change = rotate
    line_slider_x.on_change = translate_x
    line_slider_y.on_change = translate_y
    line_slider_r.on_change = change_r
    line_slider_g.on_change = change_g
    line_slider_b.on_change = change_b

    # 3. Add the sliders to the Tree2D
    # We find the node corresponding to this actor, then locate its children
    node = tree.select_node(node_name)
    translate_node = node.select_child("Translate")
    color_node = node.select_child("Color")

    # Add the translation controls
    translate_node.add_element(line_slider_x, (0.1, 0.0))
    translate_node.add_element(line_slider_y, (0.1, 1.5))
    translate_node.add_element(ring_slider, (0.6, 1.5))

    # Add the color controls
    color_node.add_element(line_slider_r, (0.1, 0.0))
    color_node.add_element(line_slider_g, (0.4, 0.0))
    color_node.add_element(line_slider_b, (0.7, 0.0))

    translate_node.set_visibility(node.expanded)
    color_node.set_visibility(node.expanded)

    # 4. Bind Actor Visibility Hooks
    def visibility_on(tree_ui):
        actor_obj.visible = True

    def visibility_off(tree_ui):
        actor_obj.visible = False

    node.on_node_select = visibility_on
    node.on_node_deselect = visibility_off


###############################################################################
# Initializing the Setup

setup_actor_controls(cube, "Cube")
setup_actor_controls(cylinder, "Cylinder")
setup_actor_controls(cone, "Cone")

###############################################################################
# Finally, we configure the scene and add both the `Tree2D` UI and our three
# actors to it.

current_size = (1000, 1000)
scene = window.Scene()
scene.add(tree, cube, cylinder, cone)

show_manager = window.ShowManager(
    scene=scene, size=current_size, title="FURY Tree2D Example"
)

show_manager.start()
