"""
==============================
Checkbox and RadioButton in UI
==============================

This example shows how to use the Checkbox and RadioButton UI elements in FURY.

Both group a small toggle button with a text label for each option and can be
laid out ``"vertical"`` (stacked) or ``"horizontal"`` (side by side):

* A **Checkbox** lets any number of options be checked at the same time.
* A **RadioButton** is mutually exclusive: exactly one option is selected, and
  choosing one clears the previous choice.

Here a vertical checkbox toggles the visibility of three colored spheres, and a
horizontal radio button selects the size shared by all of them.
"""

from fury import actor, ui, window

###############################################################################
# 1. Initialize the FURY Scene
scene = window.Scene()

###############################################################################
# 2. Create three colored sphere actors
# We keep them in a dictionary keyed by the same labels we will use for the
# checkbox options.
colors = {
    "Red": (1, 0, 0),
    "Green": (0, 1, 0),
    "Blue": (0, 0, 1),
}
spheres = {}
for i, (name, color) in enumerate(colors.items()):
    spheres[name] = actor.sphere(centers=[[i * 3 - 3, 0, 0]], colors=color, radii=1.0)
    scene.add(spheres[name])

###############################################################################
# 3. A vertical Checkbox to toggle sphere visibility
# All three options start checked, so every sphere is visible initially. Any
# combination of options can be checked at once.
checkbox = ui.Checkbox(
    labels=["Red", "Green", "Blue"],
    checked_labels=["Red", "Green", "Blue"],
    orientation="vertical",
    position=(40, 320),
    font_size=20,
)


def toggle_spheres(checkbox_ui):
    for name in colors:
        spheres[name].visible = name in checkbox_ui.checked_labels


checkbox.on_change = toggle_spheres

###############################################################################
# 4. A horizontal RadioButton to choose the size
# Only one option can be selected at a time. The only difference from the
# checkbox layout is ``orientation="horizontal"``, which lays the options out
# side by side instead of stacked.
size_dict = {
    "Small": 0.5,
    "Medium": 1.0,
    "Large": 1.6,
}
size_radio = ui.RadioButton(
    labels=["Small", "Medium", "Large"],
    checked_labels=["Medium"],
    orientation="horizontal",
    position=(40, 40),
    font_size=20,
)


def change_size(radio_ui):
    scale = size_dict[radio_ui.checked_labels[0]]
    for sphere in spheres.values():
        sphere.local.scale = (scale, scale, scale)


size_radio.on_change = change_size

###############################################################################
# 5. Add both UI elements to the scene
scene.add(checkbox)
scene.add(size_radio)

###############################################################################
# 6. Setup the ShowManager and start the event loop
showm = window.ShowManager(
    scene=scene, size=(800, 600), title="Checkbox and RadioButton Example"
)
showm.start()
