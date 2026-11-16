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
    assert tl.paused is False

    tl.seek(1)
    assert tl.paused is True

    tl.stop()
    assert tl.paused is False

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
@pytest.mark.parametrize("speed", [0.5, 2.0])
@pytest.mark.parametrize("state", ["new", "playing", "paused", "stopped", "ended"])
def test_timeline_restart_resets_actor(loop, speed, state):
    """Restart resets the real playback clock and the rendered actor."""
    box = fury.actor.box(np.array([[0.0, 0.0, 0.0]]))
    reference = fury.actor.box(np.array([[0.0, 0.0, 0.0]]))
    reference.visible = False
    animation = Animation(actors=box, loop=False)
    animation.set_position(0, [-2, 0, 0])
    animation.set_position(10, [2, 0, 0])
    timeline = Timeline(animations=animation, loop=loop)
    timeline.speed = speed

    with render_manager(box, reference) as show_m:

        def compare_frame(position, position_tolerance):
            image = render_snapshot(show_m)
            box.visible = False
            reference.local.position = position
            reference.visible = True
            expected = render_snapshot(show_m)
            reference.visible = False
            box.visible = True

            bounds = []
            for frame in (image, expected):
                y, x = np.nonzero(snapshot_mask(frame))
                assert len(x)
                bounds.append([x.min(), x.max(), y.min(), y.max()])
            # Account for real elapsed time and pixel-boundary rounding.
            pixel_tolerance = np.ceil(position_tolerance * image.shape[1] / 8) + 1
            npt.assert_allclose(bounds[0], bounds[1], rtol=0, atol=pixel_tolerance)

        if state != "new":
            timeline.play()
            timeline.seek(3)
            timeline.update()
            if state == "paused":
                timeline.pause()
            elif state == "stopped":
                timeline.stop()
            elif state == "ended":
                timeline.seek(timeline.duration)
                timeline.update()
        time.sleep(0.01)

        before_restart = time.perf_counter()
        timeline.restart()
        assert timeline.playing
        timeline.pause()
        elapsed = time.perf_counter() - before_restart
        assert 0 <= timeline.current_timestamp <= elapsed * speed + 1e-6
        assert -2 - 1e-6 <= box.local.position[0] <= -2 + 0.4 * elapsed * speed + 1e-6
        npt.assert_array_equal(box.local.position[1:], [0, 0])
        compare_frame([-2, 0, 0], 0.4 * elapsed * speed)

        paused_time = timeline.current_timestamp
        paused_position = box.local.position[0]
        timeline.play()
        time.sleep(0.01)
        before_update = timeline.current_timestamp
        timeline.update()
        timeline.pause()
        after_update = timeline.current_timestamp
        assert after_update > paused_time
        assert box.local.position[0] > paused_position
        assert -2 + 0.4 * before_update - 1e-6 <= box.local.position[0]
        assert box.local.position[0] <= -2 + 0.4 * after_update + 1e-6
        compare_frame(
            [-2 + 0.4 * after_update, 0, 0], 0.4 * (after_update - before_update)
        )
