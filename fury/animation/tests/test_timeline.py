import time
import numpy as np
import numpy.testing as npt
import fury.testing as ft
from fury.animation import Timeline
from fury.ui import PlaybackPanel


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_timeline():
    tl = Timeline(playback_panel=True)
    tl.set_position(0, np.array([1, 1, 1]))
    # overriding a keyframe
    tl.set_position(0, np.array([0, 0, 0]))
    tl.set_rotation(7, np.array([0, 180, 0]))

    # test playback panel
    ft.assert_true(isinstance(tl.playback_panel, PlaybackPanel))

    tl.update_animation()

    for t in [-10, 0, 2.2, 7, 100]:
        tl.seek(t)
        ft.assert_less_equal(tl.current_timestamp, tl.duration)
        ft.assert_greater_equal(tl.current_timestamp, 0)

        ft.assert_greater_equal(tl.current_timestamp,
                                tl.playback_panel.current_time)

        if 0 <= t <= tl.duration:
            npt.assert_almost_equal(tl.current_timestamp, t)
            # check if seeking a certain time affects the time slider's value.
            npt.assert_almost_equal(tl.current_timestamp,
                                    tl.playback_panel.current_time)

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
    tl.add_child_animation(tl_2)
    assert tl_2 in tl.child_animations

    tl_2.set_position(12, [1, 2, 1])
    assert tl_2.duration == length

    tl_2 = Timeline()
    tl_2.set_position(12, [0, 0, 1])
    assert tl_2.duration == 12

    tl = Timeline(length=12)
    tl.set_position(1, np.array([1, 1, 1]))
    assert tl.duration == 12
