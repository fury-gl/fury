"""
=========
DrawPanel
=========

This example shows how to use the DrawPanel UI. We will demonstrate how to
create Various shapes and transform them.

First, some imports.
"""
from fury import ui, window

#########################################################################
# We then create a DrawPanel Object.

drawing_canvas = ui.DrawPanel(size=(550,550),position=(25,25))

###############################################################################
# Show Manager
# ============
#
# Now we add DrawPanel to the scene.

current_size = (600, 600)
showm = window.ShowManager(size=current_size, title="DrawPanel UI Example")
showm.scene.add(drawing_canvas)
showm.start()
