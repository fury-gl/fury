import time

import numpy as np
import numpy.testing as npt
import pytest

from fury.animation import Animation, Timeline


class DummyAnimation(Animation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.updated_times = []
        self.added_scenes = []
        self.removed_scenes = []

    def update_animation(self, *, time=None):
        self.updated_times.append(time)
        self._current_timestamp = time

    def add_to_scene(self, scene):
        self.added_scenes.append(scene)

    def remove_from_scene(self, scene):
        self.removed_scenes.append(scene)


def test_timeline_creation():
    tl = Timeline()
    assert tl.duration == 0
    assert tl.loop is True
    assert tl._record_callback is None

    tl2 = Timeline(loop=False)
    assert tl2.loop is False

    tl3 = Timeline(length=10)
    tl3.update_duration()
    assert tl3.duration == 10


def test_timeline_record_requires_show_manager():
    tl = Timeline()

    with pytest.raises(RuntimeError, match="ShowManager"):
        tl.record("timeline.mp4")


def test_timeline_playback_panel():
    tl = Timeline(playback_panel=True)

    assert tl.has_playback_panel is True
    assert tl.playback_panel.on_play.__self__ is tl
    assert tl.playback_panel.on_play.__func__ is tl.play.__func__
    assert tl.playback_panel.on_stop.__self__ is tl
    assert tl.playback_panel.on_stop.__func__ is tl.stop.__func__
    assert tl.playback_panel.on_pause.__self__ is tl
    assert tl.playback_panel.on_pause.__func__ is tl.pause.__func__


def test_timeline_add_animation():
    tl = Timeline()
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([1, 1, 1]))

    tl.add_animation(anim)
    tl.add_animation(anim)

    assert tl.animations == [anim]
    assert anim.timeline is tl
    assert tl.duration == 5


def test_timeline_with_animations_param():
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([1, 1, 1]))

    anim2 = Animation()
    anim2.set_position(0, np.array([0, 0, 0]))
    anim2.set_position(8, np.array([1, 1, 1]))

    tl = Timeline(animations=[anim, anim2])

    assert tl.animations == [anim, anim2]
    assert tl.duration == 8


def test_timeline_fixed_length():
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(12, np.array([1, 1, 1]))

    tl = Timeline(animations=anim, length=8)

    assert tl.duration == 8


def test_timeline_seek():
    tl = Timeline(length=10)

    tl.seek(5)
    npt.assert_almost_equal(tl.current_timestamp, 5)

    tl.seek(15)
    npt.assert_almost_equal(tl.current_timestamp, 10)

    tl.seek(-5)
    npt.assert_almost_equal(tl.current_timestamp, 0)


def test_timeline_seek_percent():
    tl = Timeline(length=10)

    tl.seek_percent(50)
    npt.assert_almost_equal(tl.current_timestamp, 5)


def test_timeline_playback():
    tl = Timeline(length=10)

    assert tl.stopped is True
    assert tl.playing is False

    tl.play()
    assert tl.playing is True
    assert tl.stopped is False

    tl.pause()
    assert tl.paused is True
    assert tl.playing is False

    tl.stop()
    assert tl.stopped is True
    npt.assert_almost_equal(tl.current_timestamp, 0)


def test_timeline_play_updates_time():
    tl = Timeline(length=10)

    tl.play()
    t_before = tl.current_timestamp
    time.sleep(0.01)
    t_after = tl.current_timestamp

    assert t_after > t_before
    assert tl.playing is True


def test_timeline_pause_freezes_time():
    tl = Timeline(length=10)

    tl.play()
    time.sleep(0.01)
    tl.pause()

    t_before = tl.current_timestamp
    time.sleep(0.01)
    t_after = tl.current_timestamp

    npt.assert_almost_equal(t_before, t_after)


def test_timeline_speed():
    tl = Timeline(length=10)

    assert tl.speed == 1.0

    tl.speed = 2.0
    assert tl.speed == 2.0

    tl.speed = 0
    assert tl.speed == 2.0


def test_timeline_restart():
    tl = Timeline(length=10)

    tl.seek(5)
    tl.restart()

    assert tl.current_timestamp < 0.1
    assert tl.playing is True


def test_timeline_update():
    anim = DummyAnimation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([10, 10, 10]))
    tl = Timeline(animations=anim)

    tl.seek(2.5)
    tl.update(force=True)

    assert anim.updated_times[-1] == 2.5


def test_timeline_add_and_remove_from_scene():
    scene = object()
    anim = DummyAnimation()
    tl = Timeline(animations=anim)

    tl.add_to_scene(scene)
    tl.remove_from_scene(scene)

    assert anim.added_scenes == [scene]
    assert anim.removed_scenes == [scene]
