"""Test for components module."""
import itertools
import os
from os.path import join as pjoin
import shutil
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, ui, window
from fury.data import DATA_DIR
from fury.decorators import skip_osx, skip_win
from fury.primitive import prim_sphere
from fury.testing import EventCounter, assert_arrays_equal, assert_greater


def test_frame_rate_and_anti_aliasing():
    """Testing frame rate with/out anti-aliasing"""
    length_ = 200
    multi_samples = 32
    max_peels = 8

    st_x = np.arange(length_)
    st_y = np.sin(np.arange(length_))
    st_z = np.zeros(st_x.shape)
    st = np.zeros((length_, 3))
    st[:, 0] = st_x
    st[:, 1] = st_y
    st[:, 2] = st_z

    all_st = []
    all_st.append(st)
    for i in range(1000):
        all_st.append(st + i * np.array([0.0, 0.5, 0]))

    # st_actor = actor.line(all_st, linewidth=1)
    # TODO: textblock disappears when lod=True
    st_actor = actor.streamtube(all_st, linewidth=0.1, lod=False)

    scene = window.Scene()
    scene.background((1, 1.0, 1))

    # quick game style antialiasing
    scene.fxaa_on()
    scene.fxaa_off()

    # the good staff is later with multi-sampling

    tb = ui.TextBlock2D(font_size=40, color=(1, 0.5, 0))

    panel = ui.Panel2D(position=(400, 400), size=(400, 400))
    panel.add_element(tb, (0.2, 0.5))

    counter = itertools.count()
    showm = window.ShowManager(
        scene,
        size=(1980, 1080),
        reset_camera=False,
        order_transparent=True,
        multi_samples=multi_samples,
        max_peels=max_peels,
        occlusion_ratio=0.0,
    )

    scene.add(panel)
    scene.add(st_actor)
    scene.reset_camera_tight()
    scene.zoom(5)

    class FrameRateHolder:
        fpss = []

    frh = FrameRateHolder()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        if cnt % 1 == 0:
            fps = np.round(showm.frame_rate, 0)
            frh.fpss.append(fps)
            msg = 'FPS ' + str(fps) + ' ' + str(cnt)
            tb.message = msg
            showm.render()
        if cnt > 10:
            showm.exit()

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr = window.snapshot(
        scene,
        size=(1980, 1080),
        offscreen=True,
        order_transparent=True,
        multi_samples=multi_samples,
        max_peels=max_peels,
        occlusion_ratio=0.0,
    )
    assert_greater(np.sum(arr), 0)
    # TODO: check why in osx we have issues in Azure
    if not skip_osx:
        assert_greater(np.median(frh.fpss), 0)

    frh.fpss = []
    counter = itertools.count()
    multi_samples = 0
    showm = window.ShowManager(
        scene,
        size=(1980, 1080),
        reset_camera=False,
        order_transparent=True,
        multi_samples=multi_samples,
        max_peels=max_peels,
        occlusion_ratio=0.0,
    )

    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr2 = window.snapshot(
        scene,
        size=(1980, 1080),
        offscreen=True,
        order_transparent=True,
        multi_samples=multi_samples,
        max_peels=max_peels,
        occlusion_ratio=0.0,
    )
    assert_greater(np.sum(arr2), 0)
    if not skip_osx:
        assert_greater(np.median(frh.fpss), 0)


@pytest.mark.skipif(
    skip_win,
    reason='This test does not work on windows. It '
    'works on a local machine. Check after '
    'fixing memory leak with RenderWindow.',
)
def test_timer():
    """Testing add a timer and exit window and app from inside timer."""
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 50], [300, 0, 0, 100]])
    xyzr2 = np.array([[0, 200, 0, 30], [100, 200, 0, 50], [300, 200, 0, 100]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1.0, 0.45]])

    scene = window.Scene()

    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:], radii=xyzr[:, 3])

    vertices, faces = prim_sphere('repulsion724')

    sphere_actor2 = actor.sphere(
        centers=xyzr2[:, :3],
        colors=colors[:],
        radii=xyzr2[:, 3],
        vertices=vertices,
        faces=faces.astype('i8'),
    )

    scene.add(sphere_actor)
    scene.add(sphere_actor2)

    tb = ui.TextBlock2D()
    counter = itertools.count()
    showm = window.ShowManager(
        scene, size=(1024, 768), reset_camera=False, order_transparent=True
    )

    scene.add(tb)

    def timer_callback(_obj, _event):
        nonlocal counter
        cnt = next(counter)
        tb.message = "Let's count to 10 and exit :" + str(cnt)
        showm.render()
        if cnt > 9:
            showm.exit()

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr = window.snapshot(scene, offscreen=True)
    npt.assert_(np.sum(arr) > 0)
