"""
========
Button2D
========
"""

##############################################################################
# First, a bunch of imports

from fury.ui import TexturedButton2D, TextButton2D
from fury.window import (
    Scene,
    ShowManager,
)
from fury.data import fetch_viz_icons, read_viz_icons

##############################################################################
# Fetch icons that are included in FURY.

fetch_viz_icons()

##############################################################################
# Creating a Scene

scene = Scene()

#############################################################################
# Creating a Button with multiple icons

btn = TextButton2D(
    label="Hello",
    size=(100, 100),
    position=(150, 350),
    states={
        "hover": {"text": "hover", "color": (0.9, 0.9, 0.9)},
        "pressed": {"text": "pressed", "color": (0.6, 0.6, 0.6)},
        "disabled": {"text": "disabled", "color": (0.1, 0.1, 0.1)},
        "default": {"text": "default", "color": (1, 1, 1)},
    },
)
scene.add(btn)

btn = TexturedButton2D(
    states={
        "hover": read_viz_icons(fname="circle-up.png"),
        "pressed": read_viz_icons(fname="circle-down.png"),
        "disabled": read_viz_icons(fname="circle-left.png"),
        "default": read_viz_icons(fname="circle-right.png"),
    },
    size=(100, 100),
    position=(450, 350),
)
scene.add(btn)
###############################################################################
# Starting the ShowManager

if __name__ == "__main__":
    current_size = (700, 700)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Button2D Example",
    )
    show_manager.start()
