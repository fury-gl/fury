from numpy.core.fromnumeric import size
from fury import ui
from fury.ui import TreeNode2D
from fury import window


structure = [{'node-1': ['node-1.1', 'node-1.2']},
             {'node-2': ['node-2.1', 'node-2.2']},
             {'node-3': ['node-3.1', 'node-3.2', 'node-3.3']}]

tree = ui.Tree2D(structure, tree_name="Example Tree", size=(500, 500), position=(200, 200))
current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Tree2D Example")

show_manager.scene.add(tree)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()
