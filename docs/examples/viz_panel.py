"""
=======
Panel2D
=======

This example shows how to use the Panel2D.
"""

##############################################################################
# First, a bunch of imports

from fury.ui import Panel2D, Rectangle2D, Disk2D
from fury.window import (
    Scene,
    ShowManager,
)


##############################################################################
# Creating a Scene

scene = Scene()

###############################################################################
# Create a Panel2D.

panel = Panel2D(size=(300, 300), color=(0.2, 0.2, 0.2), has_border=True, border_width=5)

###############################################################################
# Let's add some simple shapes to the panel.

rect = Rectangle2D(size=(50, 50), color=(1, 0, 1))
disk = Disk2D(outer_radius=50, color=(1, 1, 0))

panel.add_element(rect, (200, 200))
panel.add_element(disk, (0.5, 0.5), anchor="center")

###############################################################################
# Now that all the elements have been initialised, we add them to the scene.
scene.add(panel)

if __name__ == "__main__":
    current_size = (800, 800)
    show_manager = ShowManager(
        scene=scene,
        size=current_size,
        title="FURY 2.0: Panel2D Example",
    )
    show_manager.start()
