"""
========
ComboBox
========

This example shows how to use the Combobox UI. We will demonstrate how to
create ComboBoxes for selecting colors for a label.

First, some imports.
"""
from fury import ui, window
from fury.data import fetch_viz_icons

##############################################################################
# First we need to fetch some icons that are included in FURY.

fetch_viz_icons()

#########################################################################
# First, we create a label.

label = ui.TextBlock2D(
    position=(200, 300),
    font_size=40,
    color=(1, 0.5, 0),
    justification='center',
    vertical_justification='top',
    text='FURY rocks!!!',
)

########################################################################
# Now we create a dictionary to store colors as its key and their
# RGB values as its value.

colors = {
    'Violet': (0.6, 0, 0.8),
    'Indigo': (0.3, 0, 0.5),
    'Blue': (0, 0, 1),
    'Green': (0, 1, 0),
    'Yellow': (1, 1, 0),
    'Orange': (1, 0.5, 0),
    'Red': (1, 0, 0),
}

########################################################################
# ComboBox
# ===================
#
# Now we create a ComboBox UI component for selecting colors.

color_combobox = ui.ComboBox2D(
    items=list(colors.keys()),
    placeholder='Choose Text Color',
    position=(75, 50),
    size=(250, 150),
)

########################################################################
# Callbacks
# ==================================
#
# Now we create a callback for setting the chosen color.


def change_color(combobox):
    label.color = colors[combobox.selected_text]


# `on_change` callback is set to `change_color` method so that
# it's called whenever a different option is selected.
color_combobox.on_change = change_color

###############################################################################
# Show Manager
# ==================================
#
# Now we add label and combobox to the scene.

current_size = (400, 400)
showm = window.ShowManager(size=current_size, title='ComboBox UI Example')
showm.scene.add(label, color_combobox)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    showm.start()

window.record(showm.scene, out_path='combobox_ui.png', size=(400, 400))
