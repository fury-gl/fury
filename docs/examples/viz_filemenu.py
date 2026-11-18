"""
================
FileMenu2D UI
================

This example demonstrates how to use the ``FileMenu2D`` UI component in FURY.
A ``FileMenu2D`` displays a file browser menu that allows navigating through
directories and selecting files. Directories are shown in green and files in
blue. Click on a directory to navigate into it, or click ``../`` to go to the
parent directory.

**Interactions:**

- **Scroll** the mouse wheel to navigate through the file list.
- **Click** a directory (green) to navigate into it.
- **Click** ``../`` to go to the parent directory.
- **Click** a file (blue) to select it.
- **Ctrl + Click** to select multiple files when ``multiselection=True``.
"""

import os

from fury import ui, window

###############################################################################
# Create the scene that will hold our UI elements.

scene = window.Scene()

###############################################################################
# Create the ``FileMenu2D`` component.
# Point it at the current directory. The component will list all subdirectories
# and files, allowing navigation through the filesystem.

current_dir = os.getcwd()

filemenu = ui.FileMenu2D(
    directory_path=current_dir,
    position=(10, 10),
    size=(300, 500),
    extensions=["*"],
    multiselection=True,
    font_size=17,
)

###############################################################################
# Optionally, attach a callback that fires whenever the user navigates to a
# new directory.


def on_directory_change():
    print(f"Navigated to: {filemenu.current_directory}")


filemenu.on_change = on_directory_change

###############################################################################
# Add the component to the scene.

scene.add(filemenu)

###############################################################################
# Display the scene. For interactive exploration, set ``interactive=True``.

interactive = False

if interactive:
    window.show(scene, size=(350, 550), title="FURY FileMenu2D Example")
else:
    window.snapshot(scene, size=(350, 550))
