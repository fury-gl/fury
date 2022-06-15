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

drawing_canvas = ui.DrawPanel(size=(550, 550), position=(25, 25))

###############################################################################
# Show Manager
# ============
#
# Now we add DrawPanel to the scene.

current_size = (600, 600)
showm = window.ShowManager(size=current_size, title="DrawPanel UI Example")

showm.scene.add(drawing_canvas)

interactive = False

if interactive:
    showm.start()
else:
    # If the UI isn't interactive, then adding a circle to the canvas
    drawing_canvas.current_mode = "circle"
    drawing_canvas.draw_shape(shape_type="circle", current_position=(275, 275))
    drawing_canvas.shape_list[-1].resize((50, 50))

    window.record(showm.scene, size=current_size,
                  out_path="viz_drawpanel.png")