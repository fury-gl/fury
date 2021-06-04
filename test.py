from fury import ui, window


###############################################################################
# Lets create multiple BulletLists and add them to a panel.
# First, let's create a panel

panel = ui.Panel2D(size=(500, 500), position=(250, 250))


###############################################################################
# Now, let's define a flat list and create a BUlletList from it


current_size = (1000, 1000)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY Bullet List Example")

show_manager.scene.add(panel)

# To interact with the UI, set interactive = True
interactive = True

if interactive:
    show_manager.start()