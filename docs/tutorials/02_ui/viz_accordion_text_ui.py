# -*- coding: utf-8 -*-
"""
================================================
Example to demonstrate the use of Accordion2D UI
================================================

This example demonstrates the usage of the Accordion UI
with text content of different sizes and lengths.
This will also demonstrate the automatic height adjustment
of the accordion UI element.

First, some imports.
"""
from fury.ui.core import TextBlock2D
from fury.ui.helpers import wrap_overflow
from fury import ui, window

###############################################################################
# Lets create a Accoridion with multiple items.
# First, let's define some varibles.

###############################################################################
# Defining the items
items = ['Small Font', 'Medium Font', 'Large Font']

###############################################################################
# Creating the respective text blocks
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do'\
       ' eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut'\
       ' enim ad minim veniam, quis nostrud exercitation ullamco laboris'\
       ' nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor'\
       ' in reprehenderit in voluptate velit esse cillum dolore eu fugiat'\
       ' nulla pariatur.'

small_text = TextBlock2D(text=text, font_size=8)
medium_text = TextBlock2D(text=text, font_size=12)
large_text = TextBlock2D(text=text, font_size=16)

textblocks = [small_text, medium_text, large_text]
###############################################################################
# Initializing the Accordion UI.
accordion = ui.elements.Accordion2D(title='FONT SIZES', items=items,
                                    body_color=(0.3, 0.5, 0.8),
                                    body_opacity=0.2, size=(300, 300),
                                    position=(0, 300),
                                    title_color=(0.8, 0.5, 0.3))

###############################################################################
# Adding the text blocks to their respective nodes.
accordion.add_content('Small Font', small_text)
accordion.add_content('Medium Font', medium_text)
accordion.add_content('Large Font', large_text)

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Accordion UI")

show_manager.scene.add(accordion)

for item, textblock in zip(items, textblocks):
    node = accordion.select_item(item)
    wrap_overflow(textblock, node.size[0])
    node.resize((node.size[0], textblock.size[1]))

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, out_path='accordion_text.png', size=current_size)
