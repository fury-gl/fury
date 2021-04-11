# -*- coding: utf-8 -*-
"""
===============
BulletList2D
===============
This example shows how to use the UI API. We will demonstrate how to create
a bullet point list.
First, some imports.
"""
from fury import ui, window


###############################################################################
# Lets create multiple BulletLists and add them to a panel.
# First, let's create a panel

panel = ui.Panel2D(size=(1000, 1000), position=(0, 0))


###############################################################################
# Now, let's define a flat list and create a BUlletList from it

flat_list = ['node-1', 'node-2', 'node-3']
flat_points = ui.BulletList2D(points=flat_list, size=(400, 400))


###############################################################################
# Now we define a nested list to see a nested BulletList
#
#
# Note that if an element is followed by a sub-list then all the elements
# in the sublist will be considered as the element's children and will have
# an indentation

nested_list = ['node-1', 'node-2', ['child-2.1', 'child2.2']]
nested_points = ui.BulletList2D(points=nested_list, size=(400, 400))


###############################################################################
# Lets create a complex BulletList.

complex_list = ['n-1', 'n-2', ['n-2.1', 'n-2.2', ['n-2.1.1']], 'n-3']
complex_points = ui.BulletList2D(points=complex_list, size=(400, 400))

# Following code demonstrates how we can add a child node in a root node
# We will add a node named n-1.1 in node n-1

complex_points.add_child_node('n-1', 'n-1.1')

###############################################################################
# Now that all lists have been defined lets add them to the panel
#
# Note that here we specify the positions with floats. In this case, these are
# percentages of the panel size.

panel.add_element(flat_points, (0, 0.45))
panel.add_element(nested_points, (0.6, 0.45))
panel.add_element(complex_points, (0, 0))


###############################################################################
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(panel)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()

window.record(show_manager.scene,
              out_path="bullet_list_ui.png", size=(1000, 1000))
