"""
=============================
FURY UI Breakdown with Tree2D
=============================

This example visualizes the different types of UI elements that are available
in FURY's UI sub-module. We will use a ``Tree2D`` UI element to organize and
display them hierarchically.

First, some imports.
"""

from fury import ui, window
from fury.ui.core import Disk2D, Rectangle2D
from fury.ui.elements import (
    LineSlider2D,
    LineDoubleSlider2D,
    RingSlider2D,
    RangeSlider,
    ListBox2D,
    ComboBox2D,
    TextButton2D,
)
from fury.ui.containers import ImageContainer2D

###############################################################################
# Defining the UI Hierarchy

structure = [
    {"Core": ["Rectangle", "Disk"]},
    {
        "Elements": [
            "Button",
            "ComboBox",
            "ListBox",
            "LineSlider",
            "LineDoubleSlider",
            "RingSlider",
            "RangeSlider",
        ]
    },
    {"Containers": ["Panels", "ImageContainers"]},
]

tree = ui.elements.Tree2D(
    structure=structure, tree_name="FURY UI Breakdown", size=(500, 500), position=(0, 0)
)

###############################################################################
# Now, let's create UI elements for the Core node.
# First, we create a Rectangle2D for the Rectangle node.

rect = Rectangle2D(size=(100, 100), color=(0.8, 0.4, 0.7))

###############################################################################
# Next, we create a Disk2D for the Disk node.

disk = Disk2D(outer_radius=50, color=(0.6, 0.2, 0.8))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content("Rectangle", rect)
tree.add_content("Disk", disk, (0.5, 0.5))

###############################################################################
# Now, let's create UI elements for the Elements node.

button = TextButton2D(label="Click Me!", size=(150, 40))
combobox = ComboBox2D(items=["Option A", "Option B", "Option C"], size=(250, 150))
listbox = ListBox2D(
    values=["First", "Second", "Third", "Fourth", "Fifth", "Sixth"], size=(250, 150)
)

lineslider = LineSlider2D(length=200, orientation="vertical")
linedoubleslider = LineDoubleSlider2D(length=200)
ringslider = RingSlider2D()
rangeslider = RangeSlider(
    range_slider_center=(0, 60), value_slider_center=(0, 0), length=200
)

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content("Button", button, (0.1, 0.5))
tree.add_content("ComboBox", combobox, (0.1, 0.5))
tree.add_content("ListBox", listbox)
tree.add_content("LineSlider", lineslider, (0.5, 0.5))
tree.add_content("LineDoubleSlider", linedoubleslider, (0.1, 0.5))
tree.add_content("RingSlider", ringslider, (0.1, 0.5))
tree.add_content("RangeSlider", rangeslider, (0.1, 0.5))

###############################################################################
# Now, we create UI elements for the Containers node.
# First, we create panels for the Panels node.

panel_first = ui.Panel2D(size=(100, 100), color=(0.5, 0.7, 0.3))
panel_second = ui.Panel2D(size=(100, 100), color=(0.3, 0.8, 0.5))

###############################################################################
# Next, we create an ImageContainer2D for the ImageContainers node.

path = (
    "https://raw.githubusercontent.com/fury-gl/"
    "fury-communication-assets/main/fury-logo.png"
)

img = ImageContainer2D(img_path=path, size=(100, 100))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content("Panels", panel_first)
tree.add_content("Panels", panel_second, (0.5, 0.5))
tree.add_content("ImageContainers", img, (0.5, 0.5))

###############################################################################
# Finally, we add the tree to our scene and start the interactive window.

current_size = (1000, 1000)

scene = window.Scene()
scene.add(tree)

show_manager = window.ShowManager(
    scene=scene, size=current_size, title="FURY Tree2D Example"
)

show_manager.start()
