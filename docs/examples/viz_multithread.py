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

def release_context(window):
    # Once release current context is available:
    # https://gitlab.kitware.com/vtk/vtk/-/merge_requests/8418
    try:
        window.ReleaseCurrent()
    except AttributeError:
        if(platform == "win32"):
            from OpenGL.WGL import wglMakeCurrent 
            wglMakeCurrent(window.GetGenericDisplayId(),None)


mutex = Lock()

xyz = 10 * np.random.rand(100, 3)
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
    mutex.acquire()
    print("Event lock acquired")
    showm.window.MakeCurrent()

def endEvent(obj, evt):
    release_context(showm.window)
    print("Event Lock Will release")
    mutex.release()
    print("Event Lock Released")


showm.initialize()

tb = ui.TextBlock2D(bold=True)


def rotate_camera():
        # Python mutex global
        global mutex
        for i in range(100):
            message = "Let's count up to 100 and exit :" + str(i+1)
            print("Getting Lock")
            mutex.acquire()
            print("Lock acquired")
            showm.window.MakeCurrent()
            tb.message = message
            print(message)
            scene.azimuth(0.05 * i)
            showm.window.Render()
            release_context(showm.window)
            print("Lock will release")
            mutex.release()
            print("Lock released")
            time.sleep(0.016)

scene.add(tb)



thread_a = Thread(target=rotate_camera)
thread_a.start()


# showm.start()
# showm.scene.AddObserver(Command.StartEvent, startEvent)
# showm.scene.AddObserver(Command.EndEvent, endEvent)

while showm.iren.GetDone() is False:
    start = time.perf_counter()
    mutex.acquire()
    showm.window.MakeCurrent()
    showm.iren.ProcessEvents()
    showm.window.Render()
    release_context(showm.window)
    mutex.release()
    end = time.perf_counter()
    # throttle to 60fps to avoid busy wait
    timePerFrame = 1.0/60.0
    if end - start < timePerFrame:
        time.sleep(timePerFrame - (end - start))
thread_a.join()


