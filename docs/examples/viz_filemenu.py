"""
================
FileMenu2D UI
================

This example demonstrates how to use the ``FileMenu2D`` UI component in
FURY. A ``FileMenu2D`` wraps a ``ListBox2D`` to browse a directory tree:
directories are listed in green, files in blue, and clicking a directory
navigates into it (or back up, via the ``../`` entry at the top).

**Interactions:**

- **Scroll** the mouse wheel to navigate through the entries.
- **Click** a directory entry (green) to move into it.
- **Click** the ``../`` entry to move up to the parent directory.
- **Click** a file entry (blue) to select it.
- **Drag** the red scroll bar on the right to jump through the list.
"""

import os
import tempfile

from fury import ui, window

###############################################################################
# Build a small, self-contained directory tree so this example doesn't
# depend on anything already on disk.

root_dir = tempfile.mkdtemp(prefix="fury_filemenu_example_")
os.makedirs(os.path.join(root_dir, "images"))
os.makedirs(os.path.join(root_dir, "data"))
for name in ("readme.txt", "script.py", "notes.txt"):
    open(os.path.join(root_dir, name), "w").close()

###############################################################################
# Create the scene that will hold our UI elements.

scene = window.Scene()

###############################################################################
# Create the ``FileMenu2D`` component, rooted at the directory built above.
# ``extensions`` restricts which files are listed; ``["*"]`` (the default)
# shows every file regardless of extension.

filemenu = ui.FileMenu2D(
    root_dir,
    extensions=["*"],
    position=(100, 100),
    size=(300, 300),
    multiselection=True,
)

scene.add(filemenu)

###############################################################################
# Set up the ``ShowManager`` and start the interactive render loop.

current_size = (600, 600)
show_manager = window.ShowManager(
    scene=scene,
    size=current_size,
    title="FURY FileMenu2D Example",
)

show_manager.start()
