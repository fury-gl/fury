from fury import ui, window
from fury.data import fetch_viz_icons
from fury.layout import GridLayout
###############################################################################
# First, let's create a panel

fetch_viz_icons()

panel = ui.Panel2D(size=(500, 500), position=(250, 250))

###############################################################################
# Now, let's add the panel to the scene


current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(panel)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()