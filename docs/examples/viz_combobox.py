"""
===================
ComboBox 2D Example
===================

This example shows how to use the ComboBox2D UI element in FURY.

A ComboBox2D allows the user to select an option from a drop-down menu.
In this example, we use it to change the color of a 3D sphere.
"""

from fury import actor, ui, window

###############################################################################
# 1. Initialize the FURY Scene
scene = window.Scene()

###############################################################################
# 2. Create a 3D Sphere Actor
sphere = actor.sphere(centers=[[1.5, 0, 0]], colors=[1, 1, 1], radii=1.0)
scene.add(sphere)

###############################################################################
# 3. Define our options mapping
# This is a dictionary mapping color names to RGB tuples.
color_dict = {
    "Red": (1, 0, 0),
    "Green": (0, 1, 0),
    "Blue": (0, 0, 1),
    "Yellow": (1, 1, 0),
    "White": (1, 1, 1),
    "Cyan": (0, 1, 1),
    "Magenta": (1, 0, 1),
    "Orange": (1, 0.5, 0),
    "Purple": (0.5, 0, 0.5),
}

###############################################################################
# 4. Initialize the ComboBox2D
# We pass the keys of our dictionary as the available items.
combobox = ui.ComboBox2D(
    items=list(color_dict.keys()),
    position=(50, 200),
    size=(250, 200),
    placeholder="Select Color...",
    draggable=True,
    font_size=20,
    selection_bg_color=(0.9, 0.9, 0.9),
    menu_opacity=0.95,
)


###############################################################################
# 5. Define a callback function
# This function will be triggered every time the user makes a new selection.
def change_color(combobox_ui):
    selected_color_name = combobox_ui.selected_text

    if selected_color_name in color_dict:
        color = color_dict[selected_color_name]
        sphere.geometry.colors.data[:] = color
        sphere.geometry.colors.update_range()


###############################################################################
# 6. Attach the callback
combobox.on_change = change_color

###############################################################################
# 7.Add the ComboBox to the scene
scene.add(combobox)

###############################################################################
# 8. Setup the ShowManager and start the event loop
showm = window.ShowManager(scene=scene, size=(800, 600), title="ComboBox2D Example")
showm.start()
