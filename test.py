from fury.ui.core import Disk2D, Rectangle2D
from fury.ui.elements import LineSlider2D, ListBox2D, TreeNode2D
from fury.ui.containers import ImageContainer2D
from fury import ui, window

structure = [{'Containers': ['Panels', 'ImageContainers']},
             {'Elements': ['ListBox', 'LineSlider']},
             {'Core': ['Rectangle', 'Disk']}]

tree = ui.elements.Tree2D(structure=structure,tree_name="FURY UI Breakdown",
                          size=(500, 500), position=(0, 0), color=(0.8, 0.4, 0.2))

###############################################################################
# Now, we create UI elements for the Containers node
# First, we create panles for the Panels node
panel = ui.Panel2D(size=(100, 100), color=(0.5, 0.7, 0.3))
panel_1 = ui.Panel2D(size=(100, 100), color=(0.3, 0.8, 0.5))

###############################################################################
# Now, we create an ImageContainer2D for the ImageContainers node
path = "https://raw.githubusercontent.com/fury-gl/"\
       "fury-communication-assets/main/fury-logo.png"

img = ImageContainer2D(img_path=path, size=(100, 100))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('Panels', panel_1)
tree.add_content('Panels', panel, (0.5, 0.5))
tree.add_content('ImageContainers', img, (0.5, 0.5))

###############################################################################
# Now, lets create UI elements for the Elements node
# First we create Listbox for the ListBox node
listbox = ListBox2D(values=['First', 'Second', 'Third', 'Fourth'])

###############################################################################
# Now, lets create a LineSlider for the LineSlider node
lineslider = LineSlider2D(length=200, orientation="vertical")

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('ListBox', listbox)
tree.add_content('LineSlider', lineslider, (0.5, 0.5))

###############################################################################
# Now, lets create UI elements for the Core node
# First we create Rectangle2D for teh Rectangle node
rect = Rectangle2D(size=(100, 100), color=(0.8, 0.4, 0.7))

###############################################################################
# Now, let's create Disk2D for the Disk node
disk = Disk2D(outer_radius=50, color=(0.6, 0.2, 0.8))

###############################################################################
# Now, we add the UI elements to their respective nodes.

tree.add_content('Rectangle', rect)
tree.add_content('Disk', disk, (0.5, 0.5))

current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(tree)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()