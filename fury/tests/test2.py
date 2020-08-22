from fury import ui, window
import os

file_dialog = ui.FileDialog2D(os.getcwd(), position=(50, 50), size=(300, 200))

def open_(file_dialog):
    print("Opening...", file_dialog.current_directory)

def close_(file_dialog):
    print("Exiting...")

file_dialog.on_accept = open_
file_dialog.on_reject = close_

sm = window.ShowManager(size=(500, 500))
sm.scene.add(file_dialog)
sm.start()