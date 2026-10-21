import time

import numpy as np
import numpy.testing as npt
import pytest

import fury
from fury.actor.tests._helpers import (
    render_manager,
    render_snapshot,
    snapshot_mask,
)
from fury.motion import Animation, Timeline


def assert_not_equal(x, y):
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


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
    updated_times = []
    anim = Animation()
    anim.add_update_callback(updated_times.append)
    anim.set_position(0, np.array([0, 0, 0]))
    anim.set_position(5, np.array([10, 10, 10]))
    tl = Timeline(animations=anim)

    tl.seek(2.5)
    tl.update(force=True)

    assert updated_times[-1] == 2.5


def test_timeline_add_and_remove_from_scene():
    scene = object()
    anim = Animation()
    tl = Timeline(animations=anim)

    tl.add_to_scene(scene)
    assert tl._scene is scene
    assert anim._scene is scene
    assert anim._added_to_scene is True

    tl.remove_from_scene(scene)

    assert tl._scene is None
    assert anim._added_to_scene is False


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("length", [6.0, 10.0])
@pytest.mark.parametrize("playback_panel", [False, True])
def test_timeline_update_applies_boundary_frame(loop, force, length, playback_panel):
    """Render a real playback boundary and continue from the resolved time."""
    box = fury.actor.box(np.array([[0.0, 0.0, 0.0]]))
    reference = fury.actor.box(np.array([[0.0, 0.0, 0.0]]))
    reference.visible = False
    animation = Animation(actors=box)
    animation.set_position(0, [-3, 0, 0])
    animation.set_position(10, [3, 0, 0])
    observed_times = []
    animation.add_update_callback(observed_times.append)
    timeline = Timeline(
        animations=animation, length=length, loop=loop, playback_panel=playback_panel
    )

    with render_manager(box, reference) as show_m:

        def compare_frame(position):
            actual = render_snapshot(show_m)
            box.visible = False
            reference.local.position = position
            reference.visible = True
            expected = render_snapshot(show_m)
            reference.visible = False
            box.visible = True
            bounds = []
            for frame in (actual, expected):
                y, x = np.nonzero(snapshot_mask(frame))
                assert len(x)
                bounds.append([x.min(), x.max(), y.min(), y.max()])
            npt.assert_allclose(bounds[0], bounds[1], rtol=0, atol=1)

        # Seek through the public API, then cross the boundary on the real clock.
        timeline.seek(length - 0.02)
        timeline.play()
        time.sleep(0.04)
        before_update = time.perf_counter()
        timeline.update(force=force)
        assert timeline.playing is loop
        expected_time = 0 if loop else length
        assert observed_times[-1] == expected_time
        if playback_panel:
            assert timeline.playback_panel.current_time == expected_time
        if loop:
            timeline.pause()
            assert (
                0
                <= timeline.current_timestamp
                <= (time.perf_counter() - before_update + 1e-6)
            )
        else:
            assert timeline.current_timestamp == length
        position = [-3 + 0.6 * expected_time, 0, 0]
        npt.assert_allclose(box.local.position, position, rtol=0, atol=1e-6)
        compare_frame(position)

        if not loop:
            count = len(observed_times)
            timeline.update()
            assert len(observed_times) == count
            timeline.update(force=True)
            assert len(observed_times) == count + 1
            assert observed_times[-1] == length

        timeline.play()
        time.sleep(0.01)
        before_time = timeline.current_timestamp
        timeline.update()
        timeline.pause()
        after_time = timeline.current_timestamp
        sample_time = observed_times[-1]
        assert 0 < before_time <= sample_time <= after_time < length
        position = [-3 + 0.6 * sample_time, 0, 0]
        npt.assert_allclose(box.local.position, position, rtol=0, atol=1e-6)
        compare_frame(position)
