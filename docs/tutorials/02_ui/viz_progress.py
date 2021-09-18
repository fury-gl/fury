# -*- coding: utf-8 -*-
"""
===========
Progress UI
===========

This example shows how to use the Progress UI. We will demonstrate how to
create a whole ProgressBar and show the progress of adding Cubes to the scene.

First, some imports.
"""
from fury import actor, ui, window
import itertools
import numpy as np

###############################################################################
# Cubes
# =====
#
# Creating List of Cubes which would be added to scene.

cube1 = actor.cube(centers=np.array([[-50, 0, 0]]),
                   colors=np.random.rand(1, 3),
                   scales=np.array([[20, 20, 20]]),
                   directions=np.array([[0, 0, 1]]))

cube2 = actor.cube(centers=np.array([[0, 0, 0]]),
                   colors=np.random.rand(1, 3),
                   scales=np.array([[20, 20, 20]]),
                   directions=np.array([[0, 0, 1]]))

cube3 = actor.cube(centers=np.array([[50, 0, 0]]),
                   colors=np.random.rand(1, 3),
                   scales=np.array([[20, 20, 20]]),
                   directions=np.array([[0, 0, 1]]))

cube_list = [cube1, cube2, cube3]

###############################################################################
# ProgressUI
# ==========

progressbar = ui.ProgressUI(position=(200, 100), padding=10, size=(400, 50),
                            initial_value=0, min_value=0,
                            max_value=len(cube_list))

###############################################################################
# Show Manager
# ============
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="FURY ProgressUI Example")

show_manager.scene.add(progressbar)

###############################################################################
# Using the on_complete hook to display a message


def remove_progressbar(ui):
    # Removing progressbar from the scene.
    for element in ui.actors:
        show_manager.scene.rm(element)


progressbar.on_complete = remove_progressbar

###############################################################################
# Creating a timer callback to add cubes to the scene after some time interval.

counter = itertools.count()


def timer_callback(_obj, _event):
    cnt = next(counter)
    show_manager.render()
    if cnt % 12 == 0 and len(cube_list):
        show_manager.scene.add(cube_list.pop(0))
        progressbar.value += 1
    if cnt == 70:
        show_manager.exit()

###############################################################################
# Initializing the ShowManager and adding callback function


show_manager.initialize()
show_manager.add_timer_callback(True, 30, timer_callback)

###############################################################################
# Set camera for better visualization

show_manager.scene.reset_camera()
show_manager.scene.set_camera(position=(0, 0, 500))
show_manager.scene.reset_clipping_range()
interactive = False

if interactive:
    show_manager.start()

window.record(show_manager.scene,
              size=current_size, out_path="viz_progress.png")
