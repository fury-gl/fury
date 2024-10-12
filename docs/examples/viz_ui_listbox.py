# -*- coding: utf-8 -*-
"""
=========
ListBox
=========

This example shows how to use the UI API. We will create a list
some geometric shapes from FURY UI elements.

First, a bunch of imports.
"""

import fury

##############################################################################
# First we need to fetch some icons that are included in FURY.

fury.data.fetch_viz_icons()

###############################################################################
# Create some text blocks that will be shown when
# list elements will be selected

welcome_text = fury.ui.TextBlock2D(text="Welcome", font_size=30, position=(500, 400))
bye_text = fury.ui.TextBlock2D(text="Bye", font_size=30, position=(500, 400))
fury_text = fury.ui.TextBlock2D(text="Fury", font_size=30, position=(500, 400))

example = [welcome_text, bye_text, fury_text]

###############################################################################
# Hide these text blocks for now


def hide_all_examples():
    for element in example:
        element.set_visibility(False)


hide_all_examples()

###############################################################################
# Create ListBox with the values as parameter.

values = ["Welcome", "Bye", "Fury"]
listbox = fury.ui.ListBox2D(
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
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = fury.window.ShowManager(
    size=current_size, title="FURY UI ListBox_Example"
)

show_manager.scene.add(listbox)
show_manager.scene.add(welcome_text)
show_manager.scene.add(bye_text)
show_manager.scene.add(fury_text)
interactive = False

if interactive:
    show_manager.start()

fury.window.record(
    scene=show_manager.scene, size=current_size, out_path="viz_listbox.png"
)
