"""Tests for fury.ui.event_recorder."""

import json
import os
import time

import numpy.testing as npt
import pygfx as gfx
import pytest

import fury.ui as ui
from fury.ui.event_recorder import (
    DEFAULT_OBSERVED_EVENTS,
    EventCounter,
    EventPlayer,
    EventRecorder,
    RecordedEvent,
)


def _make_actor():
    """Return a real pygfx Mesh with picking enabled."""
    return gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial())


def _pointer_event(event_type="pointer_down", x=100.0, y=200.0, button=1):
    """Return a real pygfx PointerEvent."""
    return gfx.PointerEvent(
        event_type,
        x=x,
        y=y,
        button=button,
        buttons=(button,),
        modifiers=(),
        time_stamp=time.perf_counter(),
    )


def _key_event(event_type="key_down", key="a", modifiers=()):
    """Return a real pygfx KeyboardEvent."""
    return gfx.KeyboardEvent(
        event_type,
        key=key,
        modifiers=modifiers,
        time_stamp=time.perf_counter(),
    )


def _wheel_event(dx=0.0, dy=-3.0):
    """Return a real pygfx WheelEvent."""
    return gfx.WheelEvent(
        "wheel",
        x=0.0,
        y=0.0,
        dx=dx,
        dy=dy,
        time_stamp=time.perf_counter(),
    )


def test_recorded_event_defaults():
    """Default fields should all be falsy/zero."""
    e = RecordedEvent("pointer_down")
    npt.assert_equal(e.event_type, "pointer_down")
    npt.assert_equal(e.x, 0.0)
    npt.assert_equal(e.y, 0.0)
    npt.assert_equal(e.button, 0)
    npt.assert_equal(e.key, "")
    npt.assert_equal(e.modifiers, ())
    npt.assert_equal(e.dx, 0.0)
    npt.assert_equal(e.dy, 0.0)


def test_recorded_event_immutable():
    """RecordedEvent should be immutable."""
    e = RecordedEvent("key_down")
    with pytest.raises(AttributeError):
        e.event_type = "pointer_down"


def test_recorded_event_from_pygfx_pointer():
    """from_pygfx_event should parse a PointerEvent correctly."""
    raw = _pointer_event("pointer_down", x=42.0, y=99.0, button=2)
    rec = RecordedEvent.from_pygfx_event(raw)
    npt.assert_equal(rec.event_type, "pointer_down")
    npt.assert_equal(rec.x, 42.0)
    npt.assert_equal(rec.y, 99.0)
    npt.assert_equal(rec.button, 2)


def test_recorded_event_from_pygfx_key():
    """from_pygfx_event should parse a KeyboardEvent correctly."""
    raw = _key_event("key_down", key="Return", modifiers=("Shift",))
    rec = RecordedEvent.from_pygfx_event(raw)
    npt.assert_equal(rec.event_type, "key_down")
    npt.assert_equal(rec.key, "Return")
    assert "Shift" in rec.modifiers


def test_recorded_event_from_pygfx_wheel():
    """from_pygfx_event should parse a WheelEvent correctly."""
    raw = _wheel_event(dx=1.0, dy=-2.5)
    rec = RecordedEvent.from_pygfx_event(raw)
    npt.assert_equal(rec.event_type, "wheel")
    npt.assert_almost_equal(rec.dx, 1.0)
    npt.assert_almost_equal(rec.dy, -2.5)


def test_recorded_event_modifiers_are_tuple():
    """Modifiers should always be stored as a tuple."""
    raw = _key_event("key_down", key="s", modifiers=("Control", "Shift"))
    rec = RecordedEvent.from_pygfx_event(raw)
    assert isinstance(rec.modifiers, tuple)
    assert "Control" in rec.modifiers
    assert "Shift" in rec.modifiers


def test_recorded_event_to_dict_round_trip():
    """to_dict and from_dict should preserve all fields."""
    original = RecordedEvent.from_pygfx_event(_wheel_event(dx=3.0, dy=-1.5))
    restored = RecordedEvent.from_dict(original.to_dict())
    npt.assert_equal(restored.event_type, original.event_type)
    npt.assert_almost_equal(restored.dx, original.dx)
    npt.assert_almost_equal(restored.dy, original.dy)
    npt.assert_equal(restored.modifiers, original.modifiers)


def test_recorded_event_from_dict_partial():
    """Minimal dict — missing fields should use defaults."""
    rec = RecordedEvent.from_dict({"event_type": "wheel"})
    npt.assert_equal(rec.x, 0.0)
    npt.assert_equal(rec.key, "")
    npt.assert_equal(rec.modifiers, ())


def test_recorded_event_to_pygfx_pointer():
    """to_pygfx_event should produce a real PointerEvent."""
    rec = RecordedEvent("pointer_down", x=10.0, y=20.0, button=1)
    evt = rec.to_pygfx_event()
    assert isinstance(evt, gfx.PointerEvent)
    npt.assert_equal(evt.type, "pointer_down")
    npt.assert_equal(evt.x, 10.0)


def test_recorded_event_to_pygfx_key():
    """to_pygfx_event should produce a real KeyboardEvent."""
    rec = RecordedEvent("key_up", key="Escape")
    evt = rec.to_pygfx_event()
    assert isinstance(evt, gfx.KeyboardEvent)
    npt.assert_equal(evt.key, "Escape")


def test_recorded_event_to_pygfx_wheel():
    """to_pygfx_event should produce a real WheelEvent."""
    rec = RecordedEvent("wheel", dy=-3.0)
    evt = rec.to_pygfx_event()
    assert isinstance(evt, gfx.WheelEvent)
    npt.assert_almost_equal(evt.dy, -3.0)


def test_event_recorder_not_recording_initially():
    """A fresh EventRecorder should not be recording."""
    recorder = EventRecorder()
    assert not recorder.is_recording


def test_event_recorder_attach_and_record():
    """EventRecorder should capture events dispatched via handle_event."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)

    actor.handle_event(_pointer_event("pointer_down", x=10.0, y=20.0))
    actor.handle_event(_pointer_event("pointer_up", x=10.0, y=20.0))

    assert recorder.is_recording
    npt.assert_equal(len(recorder.events), 2)
    npt.assert_equal(recorder.events[0].event_type, "pointer_down")
    npt.assert_equal(recorder.events[0].x, 10.0)
    npt.assert_equal(recorder.events[1].event_type, "pointer_up")
    recorder.detach()


def test_event_recorder_detach_stops_capture():
    """Events after detach() should not be captured."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    recorder.detach()
    actor.handle_event(_pointer_event("pointer_down"))
    npt.assert_equal(len(recorder.events), 1)
    assert not recorder.is_recording


def test_event_recorder_double_attach_raises():
    """Attaching twice without detaching should raise RuntimeError."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    with pytest.raises(RuntimeError, match="already attached"):
        recorder.attach(actor)
    recorder.detach()


def test_event_recorder_double_detach_safe():
    """Calling detach() twice should not raise."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    recorder.detach()
    recorder.detach()


def test_event_recorder_captures_key_event():
    """Key events should be captured with the correct key field."""
    actor = _make_actor()
    recorder = EventRecorder(observed_events=["key_down"])
    recorder.attach(actor)
    actor.handle_event(_key_event("key_down", key="Escape"))
    npt.assert_equal(recorder.events[0].key, "Escape")
    recorder.detach()


def test_event_recorder_captures_wheel_event():
    """Wheel events should be captured with dx/dy."""
    actor = _make_actor()
    recorder = EventRecorder(observed_events=["wheel"])
    recorder.attach(actor)
    actor.handle_event(_wheel_event(dy=-2.0))
    e = recorder.events[0]
    npt.assert_equal(e.event_type, "wheel")
    npt.assert_almost_equal(e.dy, -2.0)
    recorder.detach()


def test_event_recorder_captures_modifiers():
    """Modifier keys should be captured in the modifiers tuple."""
    actor = _make_actor()
    recorder = EventRecorder(observed_events=["key_down"])
    recorder.attach(actor)
    actor.handle_event(_key_event("key_down", key="s", modifiers=("Control",)))
    assert "Control" in recorder.events[0].modifiers
    recorder.detach()


def test_event_recorder_clear():
    """clear() should discard all captured events."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_move"))
    recorder.clear()
    npt.assert_equal(len(recorder.events), 0)
    recorder.detach()


def test_event_recorder_custom_observed_events():
    """Only the specified event types should be observed."""
    actor = _make_actor()
    recorder = EventRecorder(observed_events=["pointer_down"])
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    actor.handle_event(_pointer_event("pointer_up"))
    npt.assert_equal(len(recorder.events), 1)
    npt.assert_equal(recorder.events[0].event_type, "pointer_down")
    recorder.detach()


def test_event_recorder_events_returns_copy():
    """The events property should return a copy, not the internal list."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    copy = recorder.events
    copy.append("fake")
    npt.assert_equal(len(recorder.events), 1)
    recorder.detach()


def test_event_recorder_save_and_load(tmp_path):
    """save() and load() should round-trip events correctly."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down", x=5.0, y=15.0))
    actor.handle_event(_pointer_event("pointer_up", x=5.0, y=15.0))
    recorder.detach()

    p = str(tmp_path / "session.json")
    recorder.save(p)

    with open(p) as f:
        data = json.load(f)
    npt.assert_equal(data["event_count"], 2)
    npt.assert_equal(data["events"][0]["event_type"], "pointer_down")

    recorder2 = EventRecorder()
    recorder2.load(p)
    npt.assert_equal(len(recorder2.events), 2)
    npt.assert_almost_equal(recorder2.events[0].x, 5.0)
    npt.assert_equal(recorder2.events[1].event_type, "pointer_up")


def test_event_recorder_save_empty(tmp_path):
    """save() on an empty recorder should produce a valid JSON file."""
    recorder = EventRecorder()
    p = str(tmp_path / "empty.json")
    recorder.save(p)
    assert os.path.exists(p)
    with open(p) as f:
        d = json.load(f)
    npt.assert_equal(d["event_count"], 0)


def test_event_recorder_load_replaces_existing(tmp_path):
    """load() should replace any previously loaded events."""
    actor = _make_actor()
    rec1 = EventRecorder()
    rec1.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    rec1.detach()
    p = str(tmp_path / "one.json")
    rec1.save(p)

    rec2 = EventRecorder()
    rec2.attach(actor)
    actor.handle_event(_key_event("key_down"))
    rec2.detach()
    rec2.load(p)
    npt.assert_equal(len(rec2.events), 1)
    npt.assert_equal(rec2.events[0].event_type, "pointer_down")


def test_event_counter_counts_by_type():
    """get_count() should return correct per-type tallies."""
    actor = _make_actor()
    counter = EventCounter()
    counter.attach(actor)

    for _ in range(4):
        actor.handle_event(_pointer_event("pointer_down"))
    for _ in range(6):
        actor.handle_event(_pointer_event("pointer_move"))
    actor.handle_event(_key_event("key_down"))

    npt.assert_equal(counter.get_count("pointer_down"), 4)
    npt.assert_equal(counter.get_count("pointer_move"), 6)
    npt.assert_equal(counter.get_count("key_down"), 1)
    npt.assert_equal(counter.total(), 11)
    counter.detach()


def test_event_counter_unknown_returns_zero():
    """get_count() for an unseen event type should return 0."""
    npt.assert_equal(EventCounter().get_count("nonexistent"), 0)


def test_event_counter_total_initial_zero():
    """total() on a fresh counter should be 0."""
    npt.assert_equal(EventCounter().total(), 0)


def test_event_counter_counts_dict_is_copy():
    """counts() should return a copy of the internal dict."""
    actor = _make_actor()
    counter = EventCounter()
    counter.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    d = counter.counts()
    d["pointer_down"] = 9999
    npt.assert_equal(counter.get_count("pointer_down"), 1)
    counter.detach()


def test_event_counter_clear_resets_all():
    """clear() should reset both the event log and counters."""
    actor = _make_actor()
    counter = EventCounter()
    counter.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    counter.clear()
    npt.assert_equal(counter.total(), 0)
    npt.assert_equal(len(counter.events), 0)
    counter.detach()


def test_event_counter_log_populated():
    """The event log should be populated alongside counts."""
    actor = _make_actor()
    counter = EventCounter()
    counter.attach(actor)
    actor.handle_event(_pointer_event("pointer_down", x=7.0, y=8.0))
    npt.assert_equal(len(counter.events), 1)
    npt.assert_almost_equal(counter.events[0].x, 7.0)
    counter.detach()


def test_event_counter_is_subclass_of_recorder():
    """EventCounter should be a subclass of EventRecorder."""
    assert issubclass(EventCounter, EventRecorder)


def test_event_player_replays_all_events():
    """play() should dispatch all recorded events to the actor."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down", x=10.0))
    actor.handle_event(_pointer_event("pointer_up", x=10.0))
    actor.handle_event(_pointer_event("pointer_move", x=15.0))
    recorder.detach()

    received = []
    replay_actor = _make_actor()
    replay_actor.add_event_handler(
        lambda e: received.append(e.type), *DEFAULT_OBSERVED_EVENTS
    )

    player = EventPlayer(recorder=recorder, speed_factor=0.0)
    player.play(replay_actor)

    npt.assert_equal(len(received), 3)
    npt.assert_equal(received[0], "pointer_down")
    npt.assert_equal(received[1], "pointer_up")
    npt.assert_equal(received[2], "pointer_move")


def test_event_player_replays_correct_coordinates():
    """play() should dispatch events with correct x/y coordinates."""
    actor = _make_actor()
    recorder = EventRecorder(observed_events=["pointer_move"])
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_move", x=55.0, y=77.0))
    recorder.detach()

    received = []
    replay_actor = _make_actor()
    replay_actor.add_event_handler(
        lambda e: received.append((e.x, e.y)), "pointer_move"
    )

    EventPlayer(recorder=recorder, speed_factor=0.0).play(replay_actor)
    npt.assert_almost_equal(received[0][0], 55.0)
    npt.assert_almost_equal(received[0][1], 77.0)


def test_event_player_empty_safe():
    """play() on an empty recorder should not raise."""
    recorder = EventRecorder()
    EventPlayer(recorder=recorder, speed_factor=0.0).play(_make_actor())


def test_event_player_no_recorder_raises():
    """play() with no recorder should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="No events to replay"):
        EventPlayer().play(_make_actor())


def test_event_player_speed_zero_no_sleep(monkeypatch):
    """speed_factor=0 should result in no sleep() calls."""
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    for _ in range(3):
        actor.handle_event(_pointer_event("pointer_down"))
    recorder.detach()

    EventPlayer(recorder=recorder, speed_factor=0.0).play(_make_actor())
    npt.assert_equal(slept, [])


def test_event_player_load_then_play(tmp_path):
    """load() followed by play() should replay the correct events."""
    actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(actor)
    actor.handle_event(_pointer_event("pointer_down"))
    actor.handle_event(_pointer_event("pointer_up"))
    recorder.detach()

    p = str(tmp_path / "sess.json")
    recorder.save(p)

    received = []
    replay_actor = _make_actor()
    replay_actor.add_event_handler(
        lambda e: received.append(e.type), *DEFAULT_OBSERVED_EVENTS
    )

    player = EventPlayer(speed_factor=0.0)
    player.load(p)
    player.play(replay_actor)
    npt.assert_equal(len(received), 2)


def test_integration_record_save_load_replay(tmp_path):
    """Full round-trip: record -> save JSON -> load -> replay."""
    src_actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(src_actor)
    src_actor.handle_event(_pointer_event("pointer_down", x=10.0, y=20.0))
    src_actor.handle_event(_pointer_event("pointer_move", x=15.0, y=25.0))
    src_actor.handle_event(_pointer_event("pointer_up", x=15.0, y=25.0))
    recorder.detach()

    p = str(tmp_path / "round_trip.json")
    recorder.save(p)

    replayed = []
    dst_actor = _make_actor()
    dst_actor.add_event_handler(
        lambda e: replayed.append(e.type), *DEFAULT_OBSERVED_EVENTS
    )

    player = EventPlayer(speed_factor=0.0)
    player.load(p)
    player.play(dst_actor)

    npt.assert_equal(replayed, ["pointer_down", "pointer_move", "pointer_up"])


def test_integration_counter_observes_replayed(tmp_path):
    """EventCounter tallies correctly after replay into its actor."""
    src_actor = _make_actor()
    recorder = EventRecorder()
    recorder.attach(src_actor)
    src_actor.handle_event(_pointer_event("pointer_down"))
    src_actor.handle_event(_pointer_event("pointer_down"))
    src_actor.handle_event(_pointer_event("pointer_move"))
    recorder.detach()

    p = str(tmp_path / "counter.json")
    recorder.save(p)

    dst_actor = _make_actor()
    counter = EventCounter()
    counter.attach(dst_actor)

    player = EventPlayer(speed_factor=0.0)
    player.load(p)
    player.play(dst_actor)

    npt.assert_equal(counter.get_count("pointer_down"), 2)
    npt.assert_equal(counter.get_count("pointer_move"), 1)
    npt.assert_equal(counter.total(), 3)
    counter.detach()


def test_integration_wheel_round_trip(tmp_path):
    """Wheel events (dx/dy) survive save -> load -> replay."""
    src_actor = _make_actor()
    recorder = EventRecorder(observed_events=["wheel"])
    recorder.attach(src_actor)
    src_actor.handle_event(_wheel_event(dy=-3.0))
    recorder.detach()

    p = str(tmp_path / "wheel.json")
    recorder.save(p)

    received = []
    dst_actor = _make_actor()
    dst_actor.add_event_handler(lambda e: received.append(e.dy), "wheel")

    player = EventPlayer(speed_factor=0.0)
    player.load(p)
    player.play(dst_actor)

    npt.assert_almost_equal(received[0], -3.0)


def test_integration_multiple_sessions_independent(tmp_path):
    """Loading a second session replaces the first in the player."""
    src = _make_actor()

    rec1 = EventRecorder(observed_events=["pointer_down"])
    rec1.attach(src)
    for _ in range(3):
        src.handle_event(_pointer_event("pointer_down"))
    rec1.detach()
    p1 = str(tmp_path / "s1.json")
    rec1.save(p1)

    rec2 = EventRecorder(observed_events=["key_down"])
    rec2.attach(src)
    for _ in range(5):
        src.handle_event(_key_event("key_down"))
    rec2.detach()
    p2 = str(tmp_path / "s2.json")
    rec2.save(p2)

    dst = _make_actor()
    counter = EventCounter()
    counter.attach(dst)

    player = EventPlayer(speed_factor=0.0)
    player.load(p1)
    player.play(dst)
    npt.assert_equal(counter.get_count("pointer_down"), 3)

    counter.clear()
    player.load(p2)
    player.play(dst)
    npt.assert_equal(counter.get_count("key_down"), 5)
    npt.assert_equal(counter.get_count("pointer_down"), 0)
    counter.detach()


def test_recorder_attaches_to_rectangle2d():
    """EventRecorder should attach to a real Rectangle2D actor."""
    rect = ui.Rectangle2D()
    recorder = EventRecorder()
    recorder.attach(rect.actor)
    rect.actor.handle_event(_pointer_event("pointer_down", x=50.0, y=50.0))
    npt.assert_equal(len(recorder.events), 1)
    npt.assert_equal(recorder.events[0].event_type, "pointer_down")
    npt.assert_almost_equal(recorder.events[0].x, 50.0)
    recorder.detach()


def test_recorder_attaches_to_disk2d():
    """EventRecorder should attach to a real Disk2D actor."""
    disk = ui.Disk2D(outer_radius=50)
    recorder = EventRecorder()
    recorder.attach(disk.actor)
    disk.actor.handle_event(_pointer_event("pointer_down"))
    disk.actor.handle_event(_pointer_event("pointer_up"))
    npt.assert_equal(len(recorder.events), 2)
    recorder.detach()


def test_player_replays_into_rectangle2d():
    """EventPlayer should replay events into a Rectangle2D actor."""
    rect = ui.Rectangle2D()
    recorder = EventRecorder()
    recorder.attach(rect.actor)
    rect.actor.handle_event(_pointer_event("pointer_down", x=10.0, y=20.0))
    rect.actor.handle_event(_pointer_event("pointer_up", x=10.0, y=20.0))
    recorder.detach()

    received = []
    rect2 = ui.Rectangle2D()
    rect2.actor.add_event_handler(
        lambda e: received.append(e.type), *DEFAULT_OBSERVED_EVENTS
    )

    EventPlayer(recorder=recorder, speed_factor=0.0).play(rect2.actor)
    npt.assert_equal(received, ["pointer_down", "pointer_up"])


def test_counter_on_rectangle2d_click_sequence():
    """EventCounter tallies clicks on a Rectangle2D correctly."""
    rect = ui.Rectangle2D()
    counter = EventCounter()
    counter.attach(rect.actor)

    for _ in range(3):
        rect.actor.handle_event(_pointer_event("pointer_down"))
    rect.actor.handle_event(_pointer_event("pointer_up"))

    npt.assert_equal(counter.get_count("pointer_down"), 3)
    npt.assert_equal(counter.get_count("pointer_up"), 1)
    npt.assert_equal(counter.total(), 4)
    counter.detach()
