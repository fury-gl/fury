from numpy.core.fromnumeric import size
from fury import ui, window
from fury.data import read_viz_icons, fetch_viz_icons
import numpy as np
from fury.colormap import distinguishable_colormap
from fury import actor
from fury.utils import shallow_copy

###############################################################################
# First, let's create a panel

fetch_viz_icons()

panel = ui.Panel2D(size=(500, 500), position=(250, 250))

def change_color(panel):
    panel.color = (0.3, 0.6, 0.4)

actors = []
texts = []

# panel_1 = ui.Panel2D(size=(250, 250), color=(0.3, 0.6, 0.2))
# panel_2 = ui.Panel2D(size=(250, 250), color=(0.5, 0.2, 0.8))
# panel_3 = ui.Panel2D(size=(250, 250), color=(0.8, 0.4, 0.2))

# actors.append(panel_1)
# text_actor1 = actor.text_3d('panel 1', justification='center')
# texts.append(text_actor1)

# actors.append(panel_2)
# text_actor2 = actor.text_3d('panel 2', justification='center')
# texts.append(text_actor2)

# actors.append(panel_3)
# text_actor3 = actor.text_3d('panel 3', justification='center')
# texts.append(text_actor3)



# grid_ui = ui.GridUI(actors=actors, captions=texts,
#                         caption_offset=(0, -50, 0),
#                         cell_padding=(60, 60), dim=(3, 3),
#                         rotation_axis=(1, 0, 0))


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