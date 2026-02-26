"""
fury/ui/event_recorder.py

UI Event Recorder, Counter, and Player for FURY v2.

Attaches to ``show_manager.renderer`` (a rendercanvas ``EventEmitter``-backed
``Renderer``) and observes dict-based events dispatched through it.  No VTK
dependency; works with the pygfx/rendercanvas/wgpu stack used by FURY v2.

Classes
-------
RecordedEvent   – Immutable snapshot of a single rendercanvas event.
EventRecorder   – Hooks into ShowManager.renderer and records events to JSON.
EventCounter    – Subclass that tallies events by type for test assertions.
EventPlayer     – Replays a saved session into a ShowManager.renderer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Rendercanvas event types that we observe by default (lowercase strings used
# by rendercanvas._enums.EventType).
# ---------------------------------------------------------------------------
DEFAULT_OBSERVED_EVENTS: List[str] = [
    "pointer_down",
    "pointer_up",
    "pointer_move",
    "pointer_enter",
    "pointer_leave",
    "wheel",
    "key_down",
    "key_up",
    "double_click",
]


# ---------------------------------------------------------------------------
# RecordedEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedEvent:
    """Immutable snapshot of a single rendercanvas UI event.

    Attributes
    ----------
    event_type : str
        rendercanvas event type string, e.g. ``"pointer_down"``.
    timestamp : float
        Wall-clock time (``time.perf_counter()``) when captured.
    x : float
        Pointer x-coordinate in logical pixels (0 for non-pointer events).
    y : float
        Pointer y-coordinate in logical pixels (0 for non-pointer events).
    button : int
        Mouse button index (1=left, 2=right, 3=middle; 0 if none).
    buttons : int
        Bitmask of currently held mouse buttons.
    key : str
        Key symbol for keyboard events; empty string for pointer/wheel events.
    modifiers : tuple
        Active modifier key names, e.g. ``("Shift", "Control")``.
    dx : float
        Wheel delta-x (0 for non-wheel events).
    dy : float
        Wheel delta-y (0 for non-wheel events).
    raw : dict
        The original event dict for full-fidelity replay.
    """

    event_type: str
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    button: int = 0
    buttons: int = 0
    key: str = ""
    modifiers: tuple = ()
    dx: float = 0.0
    dy: float = 0.0
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        d = asdict(self)
        d["modifiers"] = list(d["modifiers"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedEvent":
        """Reconstruct from a plain dict (e.g. loaded from JSON)."""
        return cls(
            event_type=data["event_type"],
            timestamp=data.get("timestamp", 0.0),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            button=data.get("button", 0),
            buttons=data.get("buttons", 0),
            key=data.get("key", ""),
            modifiers=tuple(data.get("modifiers", [])),
            dx=data.get("dx", 0.0),
            dy=data.get("dy", 0.0),
            raw=data.get("raw", {}),
        )

    @classmethod
    def from_rendercanvas_event(cls, event: Any) -> "RecordedEvent":
        """Build a ``RecordedEvent`` from a live rendercanvas event.

        Accepts both **dict** events (from the renderer-level EventEmitter) and
        **object** events (from pygfx actor event handlers), using attribute
        access or item access transparently.

        Parameters
        ----------
        event :
            A rendercanvas event dict or event object.

        Returns
        -------
        RecordedEvent
        """
        if isinstance(event, dict):

            def _get(key: str, default: Any = None) -> Any:
                return event.get(key, default)

            raw = dict(event)
        else:

            def _get(key: str, default: Any = None) -> Any:  # type: ignore[misc]
                return getattr(event, key, default)

            # Serialise object to a dict so raw is always JSON-safe.
            try:
                raw = {
                    "event_type": getattr(
                        event, "event_type", getattr(event, "type", "")
                    ),
                    "x": float(getattr(event, "x", 0)),
                    "y": float(getattr(event, "y", 0)),
                    "button": int(getattr(event, "button", 0)),
                    "buttons": int(getattr(event, "buttons", 0)),
                    "key": str(getattr(event, "key", "")),
                    "modifiers": list(getattr(event, "modifiers", [])),
                    "dx": float(getattr(event, "dx", 0)),
                    "dy": float(getattr(event, "dy", 0)),
                    "time_stamp": float(
                        getattr(event, "time_stamp", time.perf_counter())
                    ),
                }
            except Exception:
                raw = {}

        # ``event_type`` may live under ``"event_type"`` (dict) or ``"type"``
        # attribute (some pygfx event objects).
        et = _get("event_type") or _get("type") or ""

        return cls(
            event_type=str(et),
            timestamp=float(
                ts if (ts := _get("time_stamp")) is not None else time.perf_counter()
            ),
            x=float(_get("x") or 0),
            y=float(_get("y") or 0),
            button=int(_get("button") or 0),
            buttons=int(_get("buttons") or 0),
            key=str(_get("key") or ""),
            modifiers=tuple(_get("modifiers") or []),
            dx=float(_get("dx") or 0),
            dy=float(_get("dy") or 0),
            raw=raw,
        )

    def to_rendercanvas_event(self) -> Dict[str, Any]:
        """Convert back to a rendercanvas-compatible event dict.

        If ``raw`` is populated it is used as the base so that fields not
        explicitly stored are preserved.

        Returns
        -------
        dict
        """
        if self.raw:
            evt = dict(self.raw)
        else:
            evt = {
                "event_type": self.event_type,
                "x": self.x,
                "y": self.y,
                "button": self.button,
                "buttons": self.buttons,
                "key": self.key,
                "modifiers": list(self.modifiers),
                "dx": self.dx,
                "dy": self.dy,
                "time_stamp": self.timestamp,
            }
        evt["event_type"] = self.event_type  # always authoritative
        return evt


# ---------------------------------------------------------------------------
# EventRecorder
# ---------------------------------------------------------------------------


class EventRecorder:
    """Records UI events from a FURY v2 ShowManager.

    Hooks into ``show_manager.renderer`` (the rendercanvas Renderer /
    EventEmitter) and registers an observer for every event type in
    :attr:`DEFAULT_OBSERVED_EVENTS`.  Each incoming event is packaged into a
    :class:`RecordedEvent` and appended to an internal log.

    Parameters
    ----------
    observed_events : list[str], optional
        Override the default set of observed event type strings.

    Examples
    --------
    >>> recorder = EventRecorder()
    >>> recorder.attach(show_manager)
    >>> # … user interacts …
    >>> recorder.detach()
    >>> recorder.save("session.json")
    """

    def __init__(
        self,
        observed_events: Optional[List[str]] = None,
    ) -> None:
        self._events: List[RecordedEvent] = []
        self._observed: List[str] = (
            list(observed_events)
            if observed_events is not None
            else list(DEFAULT_OBSERVED_EVENTS)
        )
        self._renderer: Any = None
        self._recording: bool = False
        # Store a stable reference to the bound method so that add/remove
        # handler identity checks (``cb is not callback``) work correctly.
        # Python bound methods are not singletons — ``self._on_event is
        # self._on_event`` can be False.
        self._callback_ref: Callable = self._on_event

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[RecordedEvent]:
        """A copy of the captured event log (read-only)."""
        return list(self._events)

    @property
    def is_recording(self) -> bool:
        """``True`` while actively recording."""
        return self._recording

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, show_manager: Any) -> None:
        """Attach to *show_manager* and begin recording events.

        Parameters
        ----------
        show_manager :
            A FURY v2 :class:`~fury.window.ShowManager`.

        Raises
        ------
        RuntimeError
            If already attached.
        AttributeError
            If a renderer cannot be resolved.
        """
        if self._recording:
            raise RuntimeError(
                "EventRecorder is already attached. Call detach() first."
            )
        renderer = self._resolve_renderer(show_manager)
        self._renderer = renderer
        # rendercanvas Renderer exposes add_event_handler (wraps EventEmitter.add_handler)
        renderer.add_event_handler(self._callback_ref, *self._observed)
        self._recording = True

    def detach(self) -> None:
        """Stop recording and remove the event observer."""
        if not self._recording:
            return
        try:
            # EventEmitter uses remove_handler; Renderer may also expose
            # remove_event_handler — try both gracefully.
            if hasattr(self._renderer, "remove_handler"):
                self._renderer.remove_handler(self._callback_ref, *self._observed)
            elif hasattr(self._renderer, "remove_event_handler"):
                self._renderer.remove_event_handler(self._callback_ref, *self._observed)
        except Exception:
            pass  # best-effort cleanup
        self._renderer = None
        self._recording = False

    def clear(self) -> None:
        """Discard all captured events."""
        self._events.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialise the event log to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path (created or overwritten).
        """
        payload = {
            "fury_event_recorder_version": "2.0",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load(self, filepath: str) -> None:
        """Load a previously saved event log, replacing any existing events.

        Parameters
        ----------
        filepath : str
            Path to a JSON file previously written by :meth:`save`.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._events = [RecordedEvent.from_dict(d) for d in payload["events"]]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_event(self, event: Any) -> None:
        """Observer callback — called for each registered event."""
        self._events.append(RecordedEvent.from_rendercanvas_event(event))

    @staticmethod
    def _resolve_renderer(show_manager: Any) -> Any:
        """Extract the rendercanvas renderer from a ShowManager.

        Parameters
        ----------
        show_manager :
            Any object exposing a ``renderer`` attribute.

        Returns
        -------
        renderer
            The renderer / EventEmitter instance.

        Raises
        ------
        AttributeError
            If no renderer can be found.
        """
        if hasattr(show_manager, "renderer"):
            return show_manager.renderer
        raise AttributeError(
            "Cannot resolve a renderer from the provided ShowManager. "
            "Expected show_manager.renderer to be a rendercanvas Renderer."
        )


# ---------------------------------------------------------------------------
# EventCounter
# ---------------------------------------------------------------------------


class EventCounter(EventRecorder):
    """Records events *and* maintains per-type counts.

    Useful when a test only needs to assert how many times a certain event
    type fired, without replaying the whole interaction.

    Examples
    --------
    >>> counter = EventCounter()
    >>> counter.attach(show_manager)
    >>> # … run test interaction …
    >>> counter.detach()
    >>> assert counter.get_count("pointer_down") == 3
    >>> assert counter.total() == 10
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_count(self, event_type: str) -> int:
        """Return how many times *event_type* was observed (0 if never).

        Parameters
        ----------
        event_type : str
            rendercanvas event type string, e.g. ``"pointer_down"``.
        """
        return self._counts.get(event_type, 0)

    def total(self) -> int:
        """Total number of events captured across all types."""
        return sum(self._counts.values())

    def counts(self) -> Dict[str, int]:
        """Copy of the full ``{event_type: count}`` mapping."""
        return dict(self._counts)

    def clear(self) -> None:
        """Reset both the event log and all counters."""
        super().clear()
        self._counts.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_event(self, event: Any) -> None:
        et = (
            event.get("event_type", "unknown")
            if isinstance(event, dict)
            else (
                getattr(event, "event_type", None) or getattr(event, "type", "unknown")
            )
        )
        self._counts[et] = self._counts.get(et, 0) + 1
        super()._on_event(event)


# ---------------------------------------------------------------------------
# EventPlayer
# ---------------------------------------------------------------------------


class EventPlayer:
    """Replays a sequence of :class:`RecordedEvent` objects into a ShowManager.

    Injects synthetic events via the renderer's ``emit`` or ``dispatch_event``
    method, making them indistinguishable from real user input.

    Parameters
    ----------
    recorder : EventRecorder, optional
        Source of events to replay.  Pass ``None`` and call :meth:`load`
        before :meth:`play`.
    speed_factor : float
        Multiplier applied to inter-event delays.
        ``1.0`` → real-time; ``0.0`` → instant (best for unit tests).
    on_event : callable, optional
        Hook called with ``(RecordedEvent, index)`` *before* each event is
        injected.  Use for inline assertions during replay.

    Examples
    --------
    >>> player = EventPlayer(speed_factor=0.0)
    >>> player.load("session.json")
    >>> player.play(show_manager)
    """

    def __init__(
        self,
        recorder: Optional[EventRecorder] = None,
        speed_factor: float = 1.0,
        on_event: Optional[Callable[[RecordedEvent, int], None]] = None,
    ) -> None:
        self._recorder = recorder
        self.speed_factor = speed_factor
        self.on_event = on_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filepath: str) -> None:
        """Load events from a JSON file written by :meth:`EventRecorder.save`.

        Parameters
        ----------
        filepath : str
            Path to the saved session JSON file.
        """
        rec = EventRecorder()
        rec.load(filepath)
        self._recorder = rec

    def play(self, show_manager: Any) -> None:
        """Replay all events into *show_manager*.

        Parameters
        ----------
        show_manager :
            A FURY v2 ShowManager whose window has been initialised.

        Raises
        ------
        RuntimeError
            If no events are available.
        """
        if self._recorder is None:
            raise RuntimeError(
                "No events to replay. Pass a recorder or call load() first."
            )

        events = self._recorder.events
        if not events:
            return

        renderer = EventRecorder._resolve_renderer(show_manager)

        # Dispatch priority:
        # 1. renderer.emit — synchronous dict-based (works in tests with mocks)
        # 2. show_manager.window._events.emit — raw EventEmitter on real FURY renderer
        # 3. renderer.dispatch_event — last resort (expects Event object, not dict)
        if hasattr(renderer, "emit"):
            _dispatch = renderer.emit
        elif hasattr(show_manager, "window") and hasattr(
            show_manager.window, "_events"
        ):
            _dispatch = show_manager.window._events.emit
        elif hasattr(renderer, "dispatch_event"):
            _dispatch = renderer.dispatch_event
        else:
            raise AttributeError("Cannot find a dict-compatible event dispatch path.")

        prev_ts: Optional[float] = None
        for idx, evt in enumerate(events):
            # Timing
            if self.speed_factor > 0 and prev_ts is not None:
                delay = (evt.timestamp - prev_ts) / self.speed_factor
                if delay > 0:
                    time.sleep(delay)
            prev_ts = evt.timestamp

            # User hook
            if self.on_event is not None:
                self.on_event(evt, idx)

            # Inject
            _dispatch(evt.to_rendercanvas_event())
