
from threading import Thread
import time

import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, window
from fury.utils import rotate, update_actor, vertices_from_actor


def test_multithreading():
    xyz = 10 * (np.random.random((100, 3)) - 0.5)
    colors = np.random.random((100, 4))
    radii = np.random.random(100) + 0.5

    scene = window.Scene()
    sphere_actor = actor.sphere(centers=xyz,
                                colors=colors,
                                radii=radii,
                                use_primitive=False)
    scene.add(sphere_actor)

    # Preparing the show manager as usual
    showm = window.ShowManager(scene,
                               size=(900, 768),
                               reset_camera=False,
                               order_transparent=True)

    # showm.initialize()

    vsa = vertices_from_actor(sphere_actor)

    def callback1():
        for i in range(100):
            if showm.lock_current():
                rotate(sphere_actor, rotation=(0.01 * i, 1, 0, 0))
                vsa[:] = 1.01 * vsa[:]
                update_actor(sphere_actor)
                showm.release_current()
                time.sleep(0.01)
            else:
                break

        showm.exit()
        # if not showm.is_done():
        #     arr = window.snapshot(scene, render_window = showm.window, fname = "test.png")
        #     showm.exit()
        #     npt.assert_equal(np.sum(arr) > 1, True)

    thread_a = Thread(target=callback1)
    thread_a.start()

    showm.start(multithreaded=True)
    thread_a.join()
