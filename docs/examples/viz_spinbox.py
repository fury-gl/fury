"""
=======
SpinBox
=======
"""
##############################################################################
# First, a bunch of imports
from fury.ui.elements import SpinBox
from fury.ui.core import TextBlock2D
from fury.window import (
    Scene,
    ShowManager,
)

##############################################################################
# Creating a Scene
scene = Scene()

##############################################################################
# Creating a basic SpinBox with default integer step
# Displays:  [ - ]  5  [ + ]
spinbox_int = SpinBox(
    position=(200, 450),
    size=(180, 45),
    min_value=0,
    max_value=10,
    initial_value=5,
    step=1,
    font_size=20,
    text_template="{value:.0f}",
)
scene.add(spinbox_int)

##############################################################################
# Creating a label to show the current integer value
label_int = TextBlock2D(
    text="Integer SpinBox (0 - 10, step 1):  5",
    font_size=16,
    color=(1, 1, 1),
    dynamic_bbox=True,
)
label_int.set_position((50, 510))
scene.add(label_int)

##############################################################################
# Wire up the callback so the label reflects the current value
def on_int_change(ui):
    label_int.message = (
        f"Integer SpinBox (0 - 10, step 1):  {int(ui.value)}"
    )

spinbox_int.on_change = on_int_change

##############################################################################
# Creating a SpinBox with a float step
spinbox_float = SpinBox(
    position=(200, 320),
    size=(200, 45),
    min_value=0.0,
    max_value=1.0,
    initial_value=0.5,
    step=0.1,
    font_size=20,
    text_template="{value:.1f}",
)
scene.add(spinbox_float)

##############################################################################
# Label for the float SpinBox
label_float = TextBlock2D(
    text="Float SpinBox (0.0 - 1.0, step 0.1):  0.5",
    font_size=16,
    color=(1, 1, 1),
    dynamic_bbox=True,
)
label_float.set_position((50, 380))
scene.add(label_float)

def on_float_change(ui):
    label_float.message = (
        f"Float SpinBox (0.0 - 1.0, step 0.1):  {ui.value:.1f}"
    )

spinbox_float.on_change = on_float_change

##############################################################################
# Creating a SpinBox with a larger step
spinbox_large = SpinBox(
    position=(200, 190),
    size=(200, 45),
    min_value=0,
    max_value=100,
    initial_value=50,
    step=10,
    font_size=20,
    text_template="{value:.0f}",
)
scene.add(spinbox_large)

##############################################################################
# Label for the large step SpinBox
label_large = TextBlock2D(
    text="Large Step SpinBox (0 - 100, step 10):  50",
    font_size=16,
    color=(1, 1, 1),
    dynamic_bbox=True,
)
label_large.set_position((50, 250))
scene.add(label_large)

def on_large_change(ui):
    label_large.message = (
        f"Large Step SpinBox (0 - 100, step 10):  {int(ui.value)}"
    )

spinbox_large.on_change = on_large_change

##############################################################################
# Starting the ShowManager
if __name__ == "__main__":
    current_size = (700, 600)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: SpinBox Example",
    )
    show_manager.start()
