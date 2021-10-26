"""
=======================================================
Multithreading Example
=======================================================

The goal of this demo is to show how to use different threads
to interact with fury. In particular, the main thread is used
to update interactions, while thread A rotates and renders the
scene.

"""

from fury import window, actor, ui
from threading import Timer, Lock, Thread
import numpy as np
import vtk
from fury.lib import Command
import time
import random
import itertools
from sys import platform



xyz = 10 * (np.random.rand(100, 3)-0.5)
colors = np.random.rand(100, 4)
radii = np.random.rand(100) + 0.5


scene = window.Scene()

sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)


scene.add(sphere_actor)

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


def startEvent(obj, evt):
    print("Getting Event lock")
    showm.lock()
    print("Event lock acquired")
    showm.window.MakeCurrent()


def endEvent(obj, evt):
    window.release_context(showm.window)
    print("Event Lock Will release")
    showm.release_lock()
    print("Event Lock Released")


showm.initialize()

tb = ui.TextBlock2D(bold=True)


def rotate_camera():
    # Python mutex global
    global mutex
    for i in range(100):
        message = "Let's count up to 100 and exit :" + str(i+1)
        showm.lock()
        showm.window.MakeCurrent()
        tb.message = message
        scene.azimuth(0.05 * i)
        showm.window.Render()
        window.release_context(showm.window)
        showm.release_lock()
        time.sleep(0.1)


scene.add(tb)

scene.ResetCamera()
# scene.zoom(0.1)

thread_a = Thread(target=rotate_camera)
thread_a.start()

showm.start(multithreaded=True)
thread_a.join()
