# -*- coding: utf-8 -*-
"""
================================================
Example to demonstrate the use of Accordion2D UI
================================================

This example demonstrates how to add a variety of UI elements to a
Accordion2D UI element. In this example we will be adding two
items to the Accordion2D UI element. The first item will be a
ImageContainer2D UI element followed by a TextBlock2D UI element.
Second item will be a TextBlock2D as bullet points and they will
be added as child nodes to the second item.

First, some imports.
"""
from fury.ui.core import TextBlock2D
from fury.ui.elements import TreeNode2D
from fury.ui.helpers import wrap_overflow
from fury import ui, window


###############################################################################
# Lets create a Accoridion with multiple items.
# First, let's define some varibles.

###############################################################################
# Defining the items, their respective icons.
items = ['What is FURY?', 'Key Features']
icons = ['https://img.icons8.com/ios/50/000000/question-mark--v1.png',
         'https://img.icons8.com/material-outlined/24/000000/features-list.png'
         ]

###############################################################################
# Defining the logo/intro text for FURY.
logo_url = 'https://raw.githubusercontent.com/fury-gl'\
           '/fury-communication-assets/main/fury-logo.png'

fury_intro_text = 'Free Unified Rendering in Python.\n'\
                  'A software library for scientific visualization in Python.'

fury_logo = ui.ImageContainer2D(img_path=logo_url)
fury_intro = ui.TextBlock2D(text=fury_intro_text)

###############################################################################
# Initializing the Accordion UI.
accordion = ui.elements.Accordion2D(title='FURY', items=items, icons=icons,
                                    body_color=(0.3, 0.5, 0.8),
                                    body_opacity=0.2, size=(300, 300),
                                    position=(0, 300),
                                    title_color=(0.8, 0.5, 0.3))

###############################################################################
# Adding the logo and intro text to the 'What is Fury' item.
accordion.add_content(items[0], fury_intro,
                      (fury_logo.size[0]+20, 0))

accordion.add_content(items[0], fury_logo)

###############################################################################
# Creating the bullet points for 'Key Features' item.
points = ['Custom User Interfaces', 'Physics Engines API', 'Custom Shaders',
          'Large amount of Tutorials and Examples']

tooltip_text = TextBlock2D()
bullet_icon = 'https://i.imgur.com/CGdn4DG.png'

points = [TreeNode2D(label=point, icon=bullet_icon, expandable=False,
                     color=(0.8, 0.5, 0.3))
          for point in points]

feature_node = accordion.select_item(items[1])

for point in points:
    feature_node.add_node(point)

###############################################################################
# Now, let's define a flat list and create a BUlletList from it


current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Accordion UI")

show_manager.scene.add(accordion)

fury_node = accordion.select_item(items[0])
wrap_overflow(fury_intro,
              fury_node.size[0]-fury_logo.size[0]-20)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene, out_path="accordion-nodes.png", size=current_size)
