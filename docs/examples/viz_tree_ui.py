# -*- coding: utf-8 -*-
"""
=============================
FURY UI Breakdown with Tree2D
=============================

This example vizualizes different types of UI elements that are available,
in FURY's UI sub-module with the help of a Tree2D UI element.

First, some imports.
"""
from fury import ui, window
from fury.ui.core import Disk2D, Rectangle2D
from fury.ui.elements import LineSlider2D, ListBox2D
from fury.ui.containers import ImageContainer2D

structure = [{'Containers': ['Panels', 'ImageContainers']},
             {'Elements': ['ListBox', 'LineSlider']},
             {'Core': ['Rectangle', 'Disk']}]

tree = ui.elements.Tree2D(structure=structure, tree_name="FURY UI Breakdown",
                          size=(500, 500), position=(0, 500),
                          color=(0.8, 0.4, 0.2))

###############################################################################
# Now, we create UI elements for the Containers node
# First, we create panles for the Panels node
panel_first = ui.Panel2D(size=(100, 100), color=(0.5, 0.7, 0.3))
panel_second = ui.Panel2D(size=(100, 100), color=(0.3, 0.8, 0.5))

###############################################################################
# Now, we create an ImageContainer2D for the ImageContainers node
path = "https://raw.githubusercontent.com/fury-gl/"\
       "fury-communication-assets/main/fury-logo.png"

img = ImageContainer2D(img_path=path, size=(100, 100))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('Panels', panel_first)
tree.add_content('Panels', panel_second, (0.5, 0.5))
tree.add_content('ImageContainers', img, (0.5, 0.5))

###############################################################################
# Now, lets create UI elements for the Elements node
# First we create Listbox for the ListBox node
listbox = ListBox2D(values=['First', 'Second', 'Third', 'Fourth'])

###############################################################################
# Now, lets create a LineSlider for the LineSlider node
lineslider = LineSlider2D(length=200, orientation="vertical")

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('ListBox', listbox)
tree.add_content('LineSlider', lineslider, (0.5, 0.5))

###############################################################################
# Now, lets create UI elements for the Core node
# First we create Rectangle2D for teh Rectangle node
rect = Rectangle2D(size=(100, 100), color=(0.8, 0.4, 0.7))

###############################################################################
# Now, let's create Disk2D for the Disk node
disk = Disk2D(outer_radius=50, color=(0.6, 0.2, 0.8))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('Rectangle', rect)
tree.add_content('Disk', disk, (0.5, 0.5))

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Tree2D Example")

show_manager.scene.add(tree)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, size=current_size,
              out_path="viz_tree2d.png")
