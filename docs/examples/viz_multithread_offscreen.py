"""
=======================================================
Multithreading Example
=======================================================

The goal of this demo is to show how to use different threads
to interact with fury. In particular, the main thread is used
to update the scene, while thread A is used to save the image.

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

def checkevent(obj, evt):
    print("Event");
    print(obj,evt)

showm.iren.AddObserver(Command.WindowResizeEvent, checkevent)
showm.window.AddObserver(Command.WindowResizeEvent, checkevent)

showm.initialize()


tb = ui.TextBlock2D(bold=True)


def save_image():
    for i in range(10):
        fig_filename = "example_%d.png"%i
        mutex.acquire()
        print("Saving %s ..."%fig_filename)
        showm.window.MakeCurrent()
        window.record(
            showm.scene, 
            size=(300, 300),
            out_path=fig_filename)
        release_context(window)
        mutex.release()
        time.sleep(random.random())


scene.add(tb)


thread_a = Thread(target=save_image, args=())
thread_a.start()

for i in range(100):
    message = "Let's count up to 100 and exit :" + str(i+1)
    mutex.acquire()
    showm.window.MakeCurrent()
    tb.message = message
    print(message)
    scene.azimuth(0.05 * i)
    release_context(showm.window)
    mutex.release()
    time.sleep(0.050)
thread_a.join()



