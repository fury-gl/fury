from fury import ui
from fury.layout import GridLayout
from fury.ui import Panel2D, Disk2D
from fury import actor, window
import numpy as np

panel = Panel2D(size=(100, 100), position=(50, 50))
panel_1 = Panel2D(size=(100, 100), position=(50, 50))

rect = GridLayout(cell_shape="rect")
rect.apply([panel_1, panel])

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(panel)
show_manager.scene.add(panel_1)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()