import time

import numpy as np
import numpy.testing as npt

from fury.animation import Animation, Timeline
import fury.testing as ft
from fury.ui import PlaybackPanel
from fury.window import Scene, ShowManager


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_timeline():
    tl = Timeline(playback_panel=True)

    # test playback panel
    ft.assert_true(isinstance(tl.playback_panel, PlaybackPanel))

    for t in [-10, 0, 2.2, 7, 100]:
        tl.seek(t)
        ft.assert_less_equal(tl.current_timestamp, tl.duration)
        ft.assert_greater_equal(tl.current_timestamp, 0)

        ft.assert_greater_equal(tl.current_timestamp, tl.playback_panel.current_time)

        if 0 <= t <= tl.duration:
            npt.assert_almost_equal(tl.current_timestamp, t)
            # check if seeking a certain time affects the time slider's value.
            npt.assert_almost_equal(
                tl.current_timestamp, tl.playback_panel.current_time
            )

    tl.play()
    t_before = tl.current_timestamp
    time.sleep(0.1)
    assert_not_equal(tl.current_timestamp, t_before)
    ft.assert_true(tl.playing)

    tl.pause()
    t_before = tl.current_timestamp
    ft.assert_true(tl.paused)
    time.sleep(0.1)
    npt.assert_almost_equal(tl.current_timestamp, t_before)

    tl.stop()
    ft.assert_true(tl.stopped)
    npt.assert_almost_equal(tl.current_timestamp, 0)

    length = 8
    tl_2 = Timeline(length=length)
    anim = Animation(length=12)
    tl_2.add_animation(anim)
    assert anim in tl_2.animations

    anim.set_position(12, [1, 2, 1])
    assert tl_2.duration == length

    tl_2 = Timeline(anim, length=11)
    assert tl_2.duration == 11

    tl = Timeline(playback_panel=True)
    assert tl.has_playback_panel is True

    tl.loop = True
    assert tl.loop is True
