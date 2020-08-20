from fury import ui, window
import os

file_dialog = ui.FileDialog2D(os.getcwd(), position=(50, 50), size=(300, 200))

sm = window.ShowManager(size=(500, 500))
sm.scene.add(file_dialog)
sm.start()