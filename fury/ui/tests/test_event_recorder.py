"""Tests for fury.ui.event_recorder.

Follows the FURY v2 test style (no display required — all mocked).
Event API matches rendercanvas EventEmitter (dict-based) + pygfx actors (object-based).

Run from repo root:
    python -m pytest fury/ui/tests/test_event_recorder.py -v
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock

import pytest

from fury.ui.event_recorder import (
    DEFAULT_OBSERVED_EVENTS,
    EventCounter,
    EventPlayer,
    EventRecorder,
    RecordedEvent,
)

# ---------------------------------------------------------------------------
# Helpers — fake rendercanvas Renderer
# ---------------------------------------------------------------------------


def _make_renderer() -> MagicMock:
    """Minimal rendercanvas Renderer mock.

    Mirrors real API surface:
        add_event_handler(callback, *types)
        remove_handler(callback, *types)
        emit(event_dict)          ← used by EventPlayer
        dispatch_event(event)     ← alternate dispatch path
    """
    renderer = MagicMock()
    _handlers: dict = {}  # type -> [(order, cb)]

    def add_event_handler(callback, *types, **kwargs):
        for t in types:
            _handlers.setdefault(t, []).append(callback)

    def remove_handler(callback, *types):
        for t in types:
            if t in _handlers:
                _handlers[t] = [cb for cb in _handlers[t] if cb is not callback]

    def emit(event_dict):
        et = event_dict.get("event_type", "")
        for cb in list(_handlers.get(et, [])) + list(_handlers.get("*", [])):
            cb(event_dict)

    renderer.add_event_handler.side_effect = add_event_handler
    renderer.remove_handler.side_effect = remove_handler
    renderer.emit.side_effect = emit
    renderer.dispatch_event.side_effect = emit  # alias for compat
    return renderer


def _make_show_manager(renderer=None) -> MagicMock:
    sm = MagicMock()
    sm.renderer = renderer or _make_renderer()
    return sm


def _ptr(et="pointer_down", x=100.0, y=200.0, button=1, ts=None):
    return {
        "event_type": et,
        "x": float(x),
        "y": float(y),
        "button": button,
        "buttons": button,
        "modifiers": [],
        "ntouches": 0,
        "time_stamp": ts if ts is not None else time.perf_counter(),
    }


def _key(et="key_down", key="a", modifiers=None, ts=None):
    return {
        "event_type": et,
        "key": key,
        "modifiers": modifiers or [],
        "time_stamp": ts if ts is not None else time.perf_counter(),
    }


def _wheel(x=0.0, y=0.0, dx=0.0, dy=-3.0, ts=None):
    return {
        "event_type": "wheel",
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "modifiers": [],
        "time_stamp": ts if ts is not None else time.perf_counter(),
    }


# ---------------------------------------------------------------------------
# RecordedEvent tests
# ---------------------------------------------------------------------------


class TestRecordedEvent:
    def test_defaults(self):
        e = RecordedEvent(event_type="pointer_down")
        assert e.event_type == "pointer_down"
        assert e.x == 0.0
        assert e.y == 0.0
        assert e.button == 0
        assert e.key == ""
        assert e.modifiers == ()
        assert e.dx == 0.0
        assert e.dy == 0.0

    def test_frozen(self):
        e = RecordedEvent(event_type="key_down")
        with pytest.raises(Exception):
            e.event_type = "pointer_down"  # type: ignore[misc]

    def test_from_dict_event(self):
        raw = _ptr("pointer_down", x=42, y=99, button=2)
        rec = RecordedEvent.from_rendercanvas_event(raw)
        assert rec.event_type == "pointer_down"
        assert rec.x == 42.0
        assert rec.y == 99.0
        assert rec.button == 2
        assert isinstance(rec.raw, dict)

    def test_from_dict_key_event(self):
        raw = _key("key_down", key="Return", modifiers=["Shift"])
        rec = RecordedEvent.from_rendercanvas_event(raw)
        assert rec.key == "Return"
        assert "Shift" in rec.modifiers

    def test_from_object_event(self):
        """from_rendercanvas_event must handle pygfx-style object events."""
        obj = MagicMock()
        obj.event_type = "pointer_down"
        obj.x = 10.0
        obj.y = 20.0
        obj.button = 1
        obj.buttons = 1
        obj.key = ""
        obj.modifiers = []
        obj.dx = 0.0
        obj.dy = 0.0
        obj.time_stamp = 1234.5
        rec = RecordedEvent.from_rendercanvas_event(obj)
        assert rec.event_type == "pointer_down"
        assert rec.x == 10.0

    def test_to_rendercanvas_event_uses_raw(self):
        raw = _ptr("pointer_down", x=10, y=20)
        rec = RecordedEvent.from_rendercanvas_event(raw)
        out = rec.to_rendercanvas_event()
        assert out["event_type"] == "pointer_down"
        assert out["x"] == 10.0

    def test_to_rendercanvas_event_without_raw(self):
        rec = RecordedEvent(event_type="pointer_up", x=5, y=6, button=1)
        out = rec.to_rendercanvas_event()
        assert out["event_type"] == "pointer_up"
        assert out["x"] == 5.0

    def test_event_type_always_correct_in_output(self):
        """Even if raw dict has stale event_type, to_rendercanvas_event fixes it."""
        raw = _ptr("pointer_down")
        rec = RecordedEvent.from_rendercanvas_event(raw)
        out = rec.to_rendercanvas_event()
        assert out["event_type"] == "pointer_down"

    def test_to_dict_round_trip(self):
        raw = _wheel(x=1, y=2, dx=3.0, dy=-1.5)
        original = RecordedEvent.from_rendercanvas_event(raw)
        restored = RecordedEvent.from_dict(original.to_dict())
        assert restored.event_type == original.event_type
        assert restored.x == original.x
        assert restored.dx == original.dx
        assert restored.dy == original.dy
        assert restored.modifiers == original.modifiers

    def test_from_dict_partial(self):
        """Minimal dict — missing fields should use defaults."""
        rec = RecordedEvent.from_dict({"event_type": "wheel"})
        assert rec.x == 0.0
        assert rec.key == ""
        assert rec.modifiers == ()

    def test_modifiers_preserved_as_tuple(self):
        raw = _key("key_down", key="s", modifiers=["Control", "Shift"])
        rec = RecordedEvent.from_rendercanvas_event(raw)
        assert isinstance(rec.modifiers, tuple)
        assert "Control" in rec.modifiers
        assert "Shift" in rec.modifiers


# ---------------------------------------------------------------------------
# EventRecorder tests
# ---------------------------------------------------------------------------


class TestEventRecorder:
    def test_attach_registers_handler(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)

        assert recorder.is_recording
        assert renderer.add_event_handler.call_count == 1
        args = renderer.add_event_handler.call_args[0]
        registered_types = set(args[1:])
        assert registered_types == set(DEFAULT_OBSERVED_EVENTS)

    def test_detach_removes_handler(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        recorder.detach()

        assert not recorder.is_recording
        assert renderer.remove_handler.call_count == 1

    def test_double_attach_raises(self):
        sm = _make_show_manager()
        recorder = EventRecorder()
        recorder.attach(sm)
        with pytest.raises(RuntimeError, match="already attached"):
            recorder.attach(sm)

    def test_double_detach_is_safe(self):
        sm = _make_show_manager()
        recorder = EventRecorder()
        recorder.attach(sm)
        recorder.detach()
        recorder.detach()  # should not raise

    def test_captures_events(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)

        renderer.emit(_ptr("pointer_down", x=10, y=20))
        renderer.emit(_ptr("pointer_up", x=10, y=20))

        assert len(recorder.events) == 2
        assert recorder.events[0].event_type == "pointer_down"
        assert recorder.events[0].x == 10.0
        assert recorder.events[1].event_type == "pointer_up"

    def test_captures_key_event(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_key("key_down", key="Escape"))
        assert recorder.events[0].key == "Escape"

    def test_captures_wheel_event(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_wheel(dy=-2.0))
        e = recorder.events[0]
        assert e.event_type == "wheel"
        assert e.dy == -2.0

    def test_captures_modifier(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_key("key_down", key="s", modifiers=["Control"]))
        assert "Control" in recorder.events[0].modifiers

    def test_no_capture_after_detach(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_ptr("pointer_down"))
        recorder.detach()
        renderer.emit(_ptr("pointer_down"))
        assert len(recorder.events) == 1

    def test_clear_resets_log(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_ptr("pointer_move"))
        recorder.clear()
        assert recorder.events == []

    def test_custom_observed_events(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder(observed_events=["pointer_down", "key_up"])
        recorder.attach(sm)
        args = renderer.add_event_handler.call_args[0]
        assert set(args[1:]) == {"pointer_down", "key_up"}

    def test_events_property_returns_copy(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_ptr("pointer_down"))
        copy = recorder.events
        copy.append("fake")
        assert len(recorder.events) == 1

    # Persistence
    def test_save_and_load(self, tmp_path):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        recorder = EventRecorder()
        recorder.attach(sm)
        renderer.emit(_ptr("pointer_down", x=5, y=15))
        renderer.emit(_ptr("pointer_up", x=5, y=15))
        recorder.detach()

        p = str(tmp_path / "session.json")
        recorder.save(p)

        with open(p) as f:
            data = json.load(f)
        assert data["event_count"] == 2
        assert data["events"][0]["event_type"] == "pointer_down"

        recorder2 = EventRecorder()
        recorder2.load(p)
        assert len(recorder2.events) == 2
        assert recorder2.events[0].x == 5.0
        assert recorder2.events[1].event_type == "pointer_up"

    def test_save_empty(self, tmp_path):
        recorder = EventRecorder()
        p = str(tmp_path / "empty.json")
        recorder.save(p)
        assert os.path.exists(p)
        with open(p) as f:
            d = json.load(f)
        assert d["event_count"] == 0

    def test_load_replaces_existing(self, tmp_path):
        rec1 = EventRecorder()
        rec1._events = [RecordedEvent.from_rendercanvas_event(_ptr("pointer_down"))]
        p = str(tmp_path / "one.json")
        rec1.save(p)

        rec2 = EventRecorder()
        rec2._events = [RecordedEvent.from_rendercanvas_event(_key("key_down"))]
        rec2.load(p)
        assert len(rec2.events) == 1
        assert rec2.events[0].event_type == "pointer_down"

    # Renderer resolution
    def test_resolve_renderer_via_attribute(self):
        renderer = MagicMock()
        sm = MagicMock(spec=["renderer"])
        sm.renderer = renderer
        assert EventRecorder._resolve_renderer(sm) is renderer

    def test_resolve_renderer_raises_for_unknown(self):
        sm = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            EventRecorder._resolve_renderer(sm)


# ---------------------------------------------------------------------------
# EventCounter tests
# ---------------------------------------------------------------------------


class TestEventCounter:
    def test_counts_by_type(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter = EventCounter()
        counter.attach(sm)

        for _ in range(4):
            renderer.emit(_ptr("pointer_down"))
        for _ in range(6):
            renderer.emit(_ptr("pointer_move"))
        renderer.emit(_key("key_down"))

        assert counter.get_count("pointer_down") == 4
        assert counter.get_count("pointer_move") == 6
        assert counter.get_count("key_down") == 1
        assert counter.total() == 11

    def test_unknown_event_returns_zero(self):
        assert EventCounter().get_count("nonexistent") == 0

    def test_total_initial_zero(self):
        assert EventCounter().total() == 0

    def test_counts_dict_is_copy(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter = EventCounter()
        counter.attach(sm)
        renderer.emit(_ptr("pointer_down"))
        d = counter.counts()
        d["pointer_down"] = 9999
        assert counter.get_count("pointer_down") == 1

    def test_clear_resets_all(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter = EventCounter()
        counter.attach(sm)
        renderer.emit(_ptr("pointer_down"))
        counter.clear()
        assert counter.total() == 0
        assert counter.events == []

    def test_full_log_populated(self):
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter = EventCounter()
        counter.attach(sm)
        renderer.emit(_ptr("pointer_down", x=7, y=8))
        assert len(counter.events) == 1
        assert counter.events[0].x == 7.0

    def test_is_subclass_of_recorder(self):
        assert issubclass(EventCounter, EventRecorder)


# ---------------------------------------------------------------------------
# EventPlayer tests
# ---------------------------------------------------------------------------


class TestEventPlayer:
    def _recorded(self, n=3, base_ts=1000.0):
        return [
            RecordedEvent.from_rendercanvas_event(
                {
                    **_ptr("pointer_down", x=float(i * 10), y=float(i * 10)),
                    "time_stamp": base_ts + i * 0.1,
                }
            )
            for i in range(n)
        ]

    def _recorder_with(self, events):
        rec = EventRecorder()
        rec._events = list(events)
        return rec

    def test_play_dispatches_all(self):
        events = self._recorded(3)
        player = EventPlayer(recorder=self._recorder_with(events), speed_factor=0.0)
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        player.play(sm)
        assert renderer.emit.call_count == 3

    def test_play_correct_event_types(self):
        events = [
            RecordedEvent.from_rendercanvas_event(_ptr("pointer_down")),
            RecordedEvent.from_rendercanvas_event(_key("key_up", key="a")),
        ]
        player = EventPlayer(recorder=self._recorder_with(events), speed_factor=0.0)
        dispatched = []
        renderer = _make_renderer()
        renderer.emit.side_effect = lambda e: dispatched.append(e["event_type"])
        sm = _make_show_manager(renderer)
        player.play(sm)
        assert dispatched == ["pointer_down", "key_up"]

    def test_play_correct_coordinates(self):
        events = [
            RecordedEvent.from_rendercanvas_event(_ptr("pointer_move", x=55, y=77))
        ]
        player = EventPlayer(recorder=self._recorder_with(events), speed_factor=0.0)
        dispatched = []
        renderer = _make_renderer()
        renderer.emit.side_effect = lambda e: dispatched.append(dict(e))
        sm = _make_show_manager(renderer)
        player.play(sm)
        assert dispatched[0]["x"] == 55.0
        assert dispatched[0]["y"] == 77.0

    def test_on_event_hook_called(self):
        events = self._recorded(4)
        calls = []
        player = EventPlayer(
            recorder=self._recorder_with(events),
            speed_factor=0.0,
            on_event=lambda evt, idx: calls.append((evt.event_type, idx)),
        )
        sm = _make_show_manager()
        player.play(sm)
        assert len(calls) == 4
        assert calls[0] == ("pointer_down", 0)
        assert calls[3] == ("pointer_down", 3)

    def test_play_empty_safe(self):
        player = EventPlayer(recorder=self._recorder_with([]), speed_factor=0.0)
        player.play(_make_show_manager())

    def test_no_recorder_raises(self):
        with pytest.raises(RuntimeError, match="No events to replay"):
            EventPlayer().play(_make_show_manager())

    def test_load_then_play(self, tmp_path):
        events = self._recorded(2)
        p = str(tmp_path / "sess.json")
        self._recorder_with(events).save(p)

        player = EventPlayer(speed_factor=0.0)
        player.load(p)
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        player.play(sm)
        assert renderer.emit.call_count == 2

    def test_speed_factor_zero_no_sleep(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        player = EventPlayer(
            recorder=self._recorder_with(self._recorded(3)), speed_factor=0.0
        )
        player.play(_make_show_manager())
        assert slept == []

    def test_speed_factor_nonzero_sleeps(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        # 3 events spaced 0.5s apart, speed_factor=2 → sleep 0.25s each gap
        events = [
            RecordedEvent.from_rendercanvas_event(
                {**_ptr("pointer_move"), "time_stamp": float(i) * 0.5}
            )
            for i in range(3)
        ]
        player = EventPlayer(recorder=self._recorder_with(events), speed_factor=2.0)
        player.play(_make_show_manager())
        assert len(slept) == 2
        for s in slept:
            assert abs(s - 0.25) < 1e-9

    def test_falls_back_to_dispatch_event(self):
        """If renderer has no emit, EventPlayer should use dispatch_event."""
        events = self._recorded(2)
        player = EventPlayer(recorder=self._recorder_with(events), speed_factor=0.0)

        renderer = MagicMock(
            spec=["add_event_handler", "remove_handler", "dispatch_event"]
        )
        dispatched = []
        renderer.dispatch_event.side_effect = lambda e: dispatched.append(
            e["event_type"]
        )
        sm = MagicMock(
            spec=["renderer"]
        )  # no 'window' attr → forces dispatch_event fallback
        sm.renderer = renderer
        player.play(sm)
        assert len(dispatched) == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_record_save_load_replay(self, tmp_path):
        """Full round-trip: record → save JSON → load → replay."""
        renderer_src = _make_renderer()
        sm_src = _make_show_manager(renderer_src)

        recorder = EventRecorder()
        recorder.attach(sm_src)
        renderer_src.emit(_ptr("pointer_down", x=10, y=20))
        renderer_src.emit(_ptr("pointer_move", x=15, y=25))
        renderer_src.emit(_ptr("pointer_up", x=15, y=25))
        recorder.detach()

        p = str(tmp_path / "round_trip.json")
        recorder.save(p)

        renderer_dst = _make_renderer()
        sm_dst = _make_show_manager(renderer_dst)
        replayed = []
        renderer_dst.emit.side_effect = lambda e: replayed.append(e["event_type"])

        player = EventPlayer(speed_factor=0.0)
        player.load(p)
        player.play(sm_dst)

        assert replayed == ["pointer_down", "pointer_move", "pointer_up"]

    def test_counter_observes_replayed(self, tmp_path):
        """EventCounter tallies correctly after replay into its ShowManager."""
        events = [
            RecordedEvent.from_rendercanvas_event(_ptr("pointer_down")),
            RecordedEvent.from_rendercanvas_event(_ptr("pointer_down")),
            RecordedEvent.from_rendercanvas_event(_ptr("pointer_move")),
        ]
        rec = EventRecorder()
        rec._events = list(events)
        p = str(tmp_path / "counter.json")
        rec.save(p)

        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter = EventCounter()
        counter.attach(sm)

        player = EventPlayer(speed_factor=0.0)
        player.load(p)
        player.play(sm)

        assert counter.get_count("pointer_down") == 2
        assert counter.get_count("pointer_move") == 1
        assert counter.total() == 3

    def test_wheel_round_trip(self, tmp_path):
        """Wheel events (dx/dy) survive save → load → replay."""
        rec = EventRecorder()
        rec._events = [RecordedEvent.from_rendercanvas_event(_wheel(dy=-3.0))]
        p = str(tmp_path / "wheel.json")
        rec.save(p)

        player = EventPlayer(speed_factor=0.0)
        player.load(p)

        dispatched = []
        renderer = _make_renderer()
        renderer.emit.side_effect = lambda e: dispatched.append(dict(e))
        sm = _make_show_manager(renderer)
        player.play(sm)

        assert dispatched[0]["event_type"] == "wheel"
        assert dispatched[0]["dy"] == -3.0

    def test_multiple_sessions_independent(self, tmp_path):
        """Loading a second session replaces the first."""

        def _make_session(fname, event_type, n):
            rec = EventRecorder()
            rec._events = [
                RecordedEvent.from_rendercanvas_event(_ptr(event_type))
                for _ in range(n)
            ]
            rec.save(fname)

        p1 = str(tmp_path / "s1.json")
        p2 = str(tmp_path / "s2.json")
        _make_session(p1, "pointer_down", 3)
        _make_session(p2, "key_down", 5)

        counter = EventCounter()
        renderer = _make_renderer()
        sm = _make_show_manager(renderer)
        counter.attach(sm)

        player = EventPlayer(speed_factor=0.0)
        player.load(p1)
        player.play(sm)
        assert counter.get_count("pointer_down") == 3

        counter.clear()
        player.load(p2)
        player.play(sm)
        assert counter.get_count("key_down") == 5
        assert counter.get_count("pointer_down") == 0
