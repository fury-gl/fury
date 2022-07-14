"""
===========
PolyLine UI
===========

This example shows how to use the Polyline UI.

First, some imports.
"""
from fury import ui, window

#########################################################################
# We then create a Polyline Object.

polyline = ui.PolyLine(points_data=[(100, 100), (100, 150),
                       (200, 200), (590, 230), (230, 50), (100, 100)])

###############################################################################
# Show Manager
# ============
#
# Now we add DrawPanel to the scene.

current_size = (600, 600)
showm = window.ShowManager(size=current_size, title="PolyLine UI Example")

showm.scene.add(polyline)

interactive = 1

if interactive:
    showm.start()

window.record(showm.scene, size=current_size,
              out_path="viz_polyline.png")
