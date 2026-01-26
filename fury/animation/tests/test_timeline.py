"""Tests for Timeline class (pygfx backend)."""

import time

import numpy as np
import numpy.testing as npt

from fury.animation import Animation, Timeline


def assert_not_equal(x, y):
    """Assert that two arrays are not equal."""
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_timeline_creation():
    """Test Timeline creation with various parameters."""
    tl = Timeline()
    assert tl.duration == 0
    assert tl.loop is True

    tl2 = Timeline(loop=False)
    assert tl2.loop is False

    tl3 = Timeline(length=10)
    tl3.update_duration()
    assert tl3.duration == 10


def test_timeline_add_animation():
    """Test adding animations to timeline."""
    tl = Timeline()
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([1, 1, 1]))

    tl.add_animation(anim)

    assert anim in tl.animations
    assert tl.duration == 5

    anim2 = Animation()
    anim2.set_position(0, np.array([0, 0, 0]))
    anim2.set_position(10, np.array([2, 2, 2]))

    tl.add_animation(anim2)
    assert anim2 in tl.animations
    assert tl.duration == 10


def test_timeline_with_animations_param():
    """Test Timeline creation with animations parameter."""
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([1, 1, 1]))

    tl = Timeline(animations=anim)
    assert anim in tl.animations
    assert tl.duration == 5

    anim2 = Animation()
    anim2.set_position(0, np.array([0, 0, 0]))
    anim2.set_position(8, np.array([1, 1, 1]))

    tl2 = Timeline(animations=[anim, anim2])
    assert len(tl2.animations) == 2
    assert tl2.duration == 8


def test_timeline_fixed_length():
    """Test Timeline with fixed length overriding animation duration."""
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(12, np.array([1, 1, 1]))

    tl = Timeline(animations=anim, length=8)
    assert tl.duration == 8


def test_timeline_seek():
    """Test Timeline seek functionality."""
    tl = Timeline(length=10)

    tl.seek(5)
    npt.assert_almost_equal(tl.current_timestamp, 5)

    tl.seek(0)
    npt.assert_almost_equal(tl.current_timestamp, 0)

    tl.seek(10)
    npt.assert_almost_equal(tl.current_timestamp, 10)

    tl.seek(15)
    npt.assert_almost_equal(tl.current_timestamp, 10)

    tl.seek(-5)
    npt.assert_almost_equal(tl.current_timestamp, 0)


def test_timeline_seek_percent():
    """Test Timeline seek_percent functionality."""
    tl = Timeline(length=10)

    tl.seek_percent(50)
    npt.assert_almost_equal(tl.current_timestamp, 5)

    tl.seek_percent(0)
    npt.assert_almost_equal(tl.current_timestamp, 0)

    tl.seek_percent(100)
    npt.assert_almost_equal(tl.current_timestamp, 10)


def test_timeline_playback():
    """Test Timeline play, pause, stop functionality."""
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
    """Test that playing timeline advances time."""
    tl = Timeline(length=10)

    tl.play()
    t_before = tl.current_timestamp
    time.sleep(0.1)
    t_after = tl.current_timestamp

    assert t_after > t_before
    assert tl.playing is True


def test_timeline_pause_freezes_time():
    """Test that pausing timeline freezes time."""
    tl = Timeline(length=10)

    tl.play()
    time.sleep(0.05)
    tl.pause()

    t_before = tl.current_timestamp
    time.sleep(0.1)
    t_after = tl.current_timestamp

    npt.assert_almost_equal(t_before, t_after)


def test_timeline_speed():
    """Test Timeline speed property."""
    tl = Timeline(length=10)

    assert tl.speed == 1.0

    tl.speed = 2.0
    assert tl.speed == 2.0

    tl.speed = 0
    assert tl.speed == 2.0  # Should not change

    tl.speed = -1
    assert tl.speed == 2.0  # Should not change


def test_timeline_loop_property():
    """Test Timeline loop property."""
    tl = Timeline(loop=True)
    assert tl.loop is True

    tl.loop = False
    assert tl.loop is False


def test_timeline_restart():
    """Test Timeline restart functionality."""
    tl = Timeline(length=10)

    tl.seek(5)
    tl.restart()

    assert tl.current_timestamp < 0.1
    assert tl.playing is True


def test_timeline_update():
    """Test Timeline update method."""
    anim = Animation()
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([10, 10, 10]))

    from fury import actor

    box = actor.box(np.array([[0, 0, 0]]))
    anim.add_actor(box)

    tl = Timeline(animations=anim)

    tl.seek(0)
    tl.update(force=True)
    npt.assert_almost_equal(box.local.position, np.array([0, 0, 0]))

    tl.seek(5)
    tl.update(force=True)
    npt.assert_almost_equal(box.local.position, np.array([10, 10, 10]))

    tl.seek(2.5)
    tl.update(force=True)
    npt.assert_almost_equal(box.local.position, np.array([5, 5, 5]))


def test_timeline_has_playback_panel():
    """Test Timeline playback panel property (not yet implemented in v2)."""
    tl = Timeline()
    assert tl.has_playback_panel is False


def test_timeline_current_timestamp_setter():
    """Test Timeline current_timestamp setter."""
    tl = Timeline(length=10)

    tl.current_timestamp = 5
    npt.assert_almost_equal(tl.current_timestamp, 5)

    tl.current_timestamp = 0
    npt.assert_almost_equal(tl.current_timestamp, 0)
