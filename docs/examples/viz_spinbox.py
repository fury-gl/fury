"""
=======
SpinBox
=======

This example demonstrates the ``SpinBox`` UI element.

Click the up/down buttons to change the value by ``step``, or click the
textbox, type a number and press Enter. The value is always clamped to
``[min_val, max_val]``.
"""

##############################################################################
# First, let's import the necessary modules.

from fury.data import fetch_viz_icons
from fury.ui import SpinBox, TextBlock2D
from fury.window import Scene, ShowManager

##############################################################################
# Fetch the bundled icon set used for the up/down buttons.

fetch_viz_icons()

##############################################################################
# Create a Scene.

scene = Scene()

##############################################################################
# Create a SpinBox with a range of [-20, 20] that moves in steps of 2.

spinbox = SpinBox(
    position=(200, 250),
    size=(300, 100),
    min_val=-20,
    max_val=20,
    initial_val=0,
    step=2,
)
scene.add(spinbox)

##############################################################################
# Show the current value in a label that updates whenever it changes.

label = TextBlock2D(
    text=f"Value: {spinbox.value}",
    position=(200, 400),
    font_size=24,
    color=(1, 1, 1),
)
scene.add(label)


def on_spinbox_change(sb):
    label.message = f"Value: {sb.value}"


spinbox.on_change = on_spinbox_change

###############################################################################
# Create and start the ShowManager.

current_size = (700, 600)
show_manager = ShowManager(
    scene=scene,
    size=current_size,
    title="FURY SpinBox Example",
)
show_manager.start()
