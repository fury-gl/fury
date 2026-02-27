"""
=========
ListBox2D
=========

This example shows how to use the ListBox2D.
"""

##############################################################################
# First, a bunch of imports

from fury.ui import ListBox2D
from fury.window import (
    Scene,
    ShowManager,
)

##############################################################################
# Creating a Scene

scene = Scene()

###############################################################################
# Create a ListBox2D with some values.

values = [
    "Option 1",
    "Option 2",
    "Option 3",
    "Option 4",
    "Option 5",
    "Option 6",
    "Option 7",
    "Option 8",
]

listbox = ListBox2D(
    values=values,
    position=(100, 100),
    size=(300, 200),
    multiselection=True,
)

###############################################################################
# Define a callback to print the current selection whenever it changes.


def on_change():
    print("Selected:", listbox.selected)


listbox.on_change = on_change

###############################################################################
# Now that all the elements have been initialised, we add them to the scene.

scene.add(listbox)

if __name__ == "__main__":
    current_size = (600, 500)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY ListBox2D Example",
    )
    show_manager.start()
