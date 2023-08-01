"""
===========
File Dialog
===========

This example shows how to use the File Dialog UI. We will demonstrate how to
create File Dialogs for browsing the file system and to get the desired file
path.

First, some imports.
"""

import os

from fury import ui, window

########################################################################
# File Dialog
# ================
#
# We create a couple of File Dialogs, one for saving a file and,
# the other for opening a file.
# We also create a TextBlock to display the filenames.

file_dialog_save = ui.FileDialog2D(os.getcwd(), position=(25, 25),
                                   size=(300, 200),
                                   dialog_type="Save")
file_dialog_open = ui.FileDialog2D(os.getcwd(), position=(180, 250),
                                   size=(300, 200),
                                   dialog_type="Open")

tb = ui.TextBlock2D(text="", position=(100, 300))


########################################################################
# Callbacks
# ==================================
#
# Now we create a callback which triggers when a specific action like,
# open, save or close is performed.

def open_(file_dialog):
    tb.message = "File:" + file_dialog.current_file


def save_(file_dialog):
    tb.message = "File:" + file_dialog.current_file
    tb.message += "\nSave File:" + file_dialog.save_filename


def close_(file_dialog):
    tb.message = "Exiting..."


# Callbacks are assigned to specific methods.

file_dialog_save.on_accept = save_
file_dialog_save.on_reject = close_

file_dialog_open.on_accept = open_
file_dialog_open.on_reject = close_

###############################################################################
# Show Manager
# ==================================
#
# Now we add the File Dialogs and the TextBlock to the scene.

sm = window.ShowManager(size=(500, 500))
sm.scene.add(file_dialog_open, file_dialog_save, tb)

# To interact with the UI, set interactive = True
interactive = False

if interactive:
    sm.start()

window.record(sm.scene, out_path="viz_file_dialog.png", size=(500, 500))
