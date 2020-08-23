from fury import ui, window
import os

file_dialog = ui.FileDialog2D(os.getcwd(), position=(50, 50), size=(300, 200), dialog_type="Save")
tb = ui.TextBlock2D(text="", position=(100, 300))

def open_(file_dialog):
    tb.message = "File:" + file_dialog.current_file
    tb.message += "\nSave File:" + file_dialog.save_filename

def close_(file_dialog):
    tb.message = "Exiting..."

file_dialog.on_accept = open_
file_dialog.on_reject = close_

sm = window.ShowManager(size=(500, 500))
sm.scene.add(file_dialog, tb)
sm.start()