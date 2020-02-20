# -*- coding: utf-8 -*-
"""
=========
ListBox
=========

This example shows how to use the UI API. We will create a list
some geometric shapes from DIPY UI elements.

First, a bunch of imports.
"""
from fury import ui, window

###############################################################################
# Create some text blocks that will be showm when
# list elements will be selected

Get_text = ui.TextBlock2D(text="Get", font_size=30, position=(500, 400))
Set_text = ui.TextBlock2D(text="Set", font_size=30, position=(500, 400))
Fury_text = ui.TextBlock2D(text="Fury", font_size=30, position=(500, 400))

example = [Get_text, Set_text, Fury_text]

###############################################################################
# Hide these text blocks for now


def hide_all_examples():
    for element in example:
        element.set_visibility(False)


hide_all_examples()

###############################################################################
# Create ListBox with the values as parameter.

values = ["Get", "Set", "Fury"]
listbox = ui.ListBox2D(
    values=values, position=(10, 300), size=(200, 200), multiselection=False
)

###############################################################################
# Function to show selected element.


def display_element():
    hide_all_examples()
    element = example[values.index(listbox.selected[0])]
    element.set_visibility(True)


listbox.on_change = display_element

###############################################################################
# Show Manager
# ==================================
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size, 
                                  title="DIPY UI ListBox_Example")

show_manager.scene.add(listbox)
show_manager.scene.add(Get_text)
show_manager.scene.add(Set_text)
show_manager.scene.add(Fury_text)
show_manager.start()
