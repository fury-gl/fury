"""UI event recorder, counter, and player for FURY v2.

Attaches to a pygfx WorldObject actor and observes events dispatched
through it. Works with the pygfx/rendercanvas/wgpu stack used by FURY v2.
"""

import json
import time

DEFAULT_OBSERVED_EVENTS = [
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


class RecordedEvent:
    """Immutable snapshot of a single pygfx UI event.

    Parameters
    ----------
    event_type : str
        Pygfx event type string, e.g. ``"pointer_down"``.
    timestamp : float, optional
        Wall-clock time when captured.
    x : float, optional
        Pointer x-coordinate in logical pixels.
    y : float, optional
        Pointer y-coordinate in logical pixels.
    button : int, optional
        Mouse button index (1=left, 2=right, 3=middle; 0 if none).
    buttons : tuple, optional
        Currently held mouse buttons.
    key : str, optional
        Key symbol for keyboard events.
    modifiers : tuple, optional
        Active modifier key names, e.g. ``("Shift", "Control")``.
    dx : float, optional
        Wheel delta-x.
    dy : float, optional
        Wheel delta-y.
    """

    def __init__(
        self,
        event_type,
        *,
        timestamp=0.0,
        x=0.0,
        y=0.0,
        button=0,
        buttons=(),
        key="",
        modifiers=(),
        dx=0.0,
        dy=0.0,
    ):
        """Initialise a RecordedEvent.

        Parameters
        ----------
        event_type : str
            Pygfx event type string, e.g. ``"pointer_down"``.
        timestamp : float, optional
            Wall-clock time when captured.
        x : float, optional
            Pointer x-coordinate in logical pixels.
        y : float, optional
            Pointer y-coordinate in logical pixels.
        button : int, optional
            Mouse button index (1=left, 2=right, 3=middle; 0 if none).
        buttons : tuple, optional
            Currently held mouse buttons.
        key : str, optional
            Key symbol for keyboard events.
        modifiers : tuple, optional
            Active modifier key names.
        dx : float, optional
            Wheel delta-x.
        dy : float, optional
            Wheel delta-y.
        """
        self._event_type = str(event_type)
        self._timestamp = float(timestamp)
        self._x = float(x)
        self._y = float(y)
        self._button = int(button)
        self._buttons = tuple(buttons)
        self._key = str(key)
        self._modifiers = tuple(modifiers)
        self._dx = float(dx)
        self._dy = float(dy)

    @property
    def event_type(self):
        """Pygfx event type string.

        Returns
        -------
        str
            Pygfx event type string, e.g. ``"pointer_down"``.
        """
        return self._event_type

    @property
    def timestamp(self):
        """Wall-clock time when captured.

        Returns
        -------
        float
            Wall-clock time when captured.
        """
        return self._timestamp

    @property
    def x(self):
        """Pointer x-coordinate.

        Returns
        -------
        float
            Pointer x-coordinate in logical pixels.
        """
        return self._x

    @property
    def y(self):
        """Pointer y-coordinate.

        Returns
        -------
        float
            Pointer y-coordinate in logical pixels.
        """
        return self._y

    @property
    def button(self):
        """Mouse button index.

        Returns
        -------
        int
            Mouse button index (1=left, 2=right, 3=middle; 0 if none).
        """
        return self._button

    @property
    def buttons(self):
        """Currently held mouse buttons.

        Returns
        -------
        tuple
            Currently held mouse buttons.
        """
        return self._buttons

    @property
    def key(self):
        """Key symbol for keyboard events.

        Returns
        -------
        str
            Key symbol for keyboard events.
        """
        return self._key

    @property
    def modifiers(self):
        """Active modifier key names.

        Returns
        -------
        tuple
            Active modifier key names.
        """
        return self._modifiers

    @property
    def dx(self):
        """Wheel delta-x.

        Returns
        -------
        float
            Wheel delta-x.
        """
        return self._dx

    @property
    def dy(self):
        """Wheel delta-y.

        Returns
        -------
        float
            Wheel delta-y.
        """
        return self._dy

    def __setattr__(self, name, value):
        """Prevent mutation of public fields after construction.

        Parameters
        ----------
        name : str
            Attribute name.
        value : object
            Attribute value.

        Raises
        ------
        AttributeError
            If attempting to set a public (non-private) attribute.
        """
        if not name.startswith("_"):
            raise AttributeError("RecordedEvent is immutable.")
        super().__setattr__(name, value)

    def to_dict(self):
        """Return a JSON-serialisable representation.

        Returns
        -------
        dict
            JSON-serialisable dict of all event fields.
        """
        return {
            "event_type": self._event_type,
            "timestamp": self._timestamp,
            "x": self._x,
            "y": self._y,
            "button": self._button,
            "buttons": list(self._buttons),
            "key": self._key,
            "modifiers": list(self._modifiers),
            "dx": self._dx,
            "dy": self._dy,
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from a plain dict (e.g. loaded from JSON).

        Parameters
        ----------
        data : dict
            Plain dict with event fields.

        Returns
        -------
        RecordedEvent
            Reconstructed event instance.
        """
        return cls(
            data["event_type"],
            timestamp=data.get("timestamp", 0.0),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            button=data.get("button", 0),
            buttons=tuple(data.get("buttons", ())),
            key=data.get("key", ""),
            modifiers=tuple(data.get("modifiers", ())),
            dx=data.get("dx", 0.0),
            dy=data.get("dy", 0.0),
        )

    @classmethod
    def from_pygfx_event(cls, event):
        """Build a RecordedEvent from a live pygfx event object.

        Parameters
        ----------
        event : pygfx Event
            A pygfx PointerEvent, KeyboardEvent, WheelEvent or similar.

        Returns
        -------
        RecordedEvent
            Snapshot of the given event.
        """
        et = str(getattr(event, "type", "") or "")
        ts = getattr(event, "_time_stamp", None) or time.perf_counter()
        return cls(
            et,
            timestamp=float(ts),
            x=float(getattr(event, "x", 0) or 0),
            y=float(getattr(event, "y", 0) or 0),
            button=int(getattr(event, "button", 0) or 0),
            buttons=tuple(getattr(event, "buttons", ()) or ()),
            key=str(getattr(event, "key", "") or ""),
            modifiers=tuple(getattr(event, "modifiers", ()) or ()),
            dx=float(getattr(event, "dx", 0) or 0),
            dy=float(getattr(event, "dy", 0) or 0),
        )

    def to_pygfx_event(self):
        """Convert back to a pygfx-compatible event object.

        Returns
        -------
        pygfx Event
            A live pygfx event that can be passed to handle_event.
        """
        import pygfx as gfx

        et = self._event_type
        if et in (
            "pointer_down",
            "pointer_up",
            "pointer_move",
            "pointer_enter",
            "pointer_leave",
            "double_click",
            "click",
        ):
            return gfx.PointerEvent(
                et,
                x=self._x,
                y=self._y,
                button=self._button,
                buttons=self._buttons,
                modifiers=self._modifiers,
                time_stamp=self._timestamp,
            )
        elif et in ("key_down", "key_up"):
            return gfx.KeyboardEvent(
                et,
                key=self._key,
                modifiers=self._modifiers,
                time_stamp=self._timestamp,
            )
        elif et == "wheel":
            return gfx.WheelEvent(
                et,
                x=0.0,
                y=0.0,
                dx=self._dx,
                dy=self._dy,
                time_stamp=self._timestamp,
            )
        else:
            return gfx.Event(et, time_stamp=self._timestamp)


class EventRecorder:
    """Records UI events from a pygfx WorldObject actor.

    Attaches to any pygfx actor that exposes ``add_event_handler`` and
    ``remove_event_handler``, capturing each event as a
    :class:`RecordedEvent`.

    Parameters
    ----------
    observed_events : list, optional
        Override the default set of observed event type strings.

    Examples
    --------
    >>> recorder = EventRecorder()
    >>> recorder.is_recording
    False
    """

    def __init__(self, observed_events=None):
        """Initialise EventRecorder.

        Parameters
        ----------
        observed_events : list, optional
            Override the default set of observed event type strings.
        """
        self._events = []
        self._observed = (
            list(observed_events)
            if observed_events is not None
            else list(DEFAULT_OBSERVED_EVENTS)
        )
        self._actor = None
        self._recording = False

    @property
    def events(self):
        """A copy of the captured event log.

        Returns
        -------
        list
            Copy of the internal event log.
        """
        return list(self._events)

    @property
    def is_recording(self):
        """True while actively recording.

        Returns
        -------
        bool
            Whether the recorder is currently attached and recording.
        """
        return self._recording

    def attach(self, actor):
        """Attach to a pygfx actor and begin recording events.

        Parameters
        ----------
        actor : pygfx WorldObject
            Any pygfx actor with ``add_event_handler`` / ``remove_event_handler``.

        Raises
        ------
        RuntimeError
            If already attached.
        """
        if self._recording:
            raise RuntimeError(
                "EventRecorder is already attached. Call detach() first."
            )
        actor.add_event_handler(self._on_event, *self._observed)
        self._actor = actor
        self._recording = True

    def detach(self):
        """Stop recording and remove the event observer."""
        if not self._recording:
            return
        try:
            self._actor.remove_event_handler(self._on_event, *self._observed)
        except Exception:
            pass
        self._actor = None
        self._recording = False

    def clear(self):
        """Discard all captured events."""
        self._events.clear()

    def save(self, filepath):
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

    def load(self, filepath):
        """Load a previously saved event log, replacing any existing events.

        Parameters
        ----------
        filepath : str
            Path to a JSON file previously written by :meth:`save`.
        """
        with open(filepath, encoding="utf-8") as fh:
            payload = json.load(fh)
        self._events = [RecordedEvent.from_dict(d) for d in payload["events"]]

    def _on_event(self, event):
        """Append a captured event to the internal log.

        Parameters
        ----------
        event : pygfx Event
            A live pygfx event object.
        """
        self._events.append(RecordedEvent.from_pygfx_event(event))


class EventCounter(EventRecorder):
    """Records events and maintains per-type counts.

    Useful when a test only needs to assert how many times a certain
    event type fired, without replaying the whole interaction.

    Parameters
    ----------
    observed_events : list, optional
        Override the default set of observed event type strings.

    Examples
    --------
    >>> counter = EventCounter()
    >>> counter.total()
    0
    """

    def __init__(self, observed_events=None):
        """Initialise EventCounter.

        Parameters
        ----------
        observed_events : list, optional
            Override the default set of observed event type strings.
        """
        super().__init__(observed_events=observed_events)
        self._counts = {}

    def get_count(self, event_type):
        """Return how many times event_type was observed.

        Parameters
        ----------
        event_type : str
            Pygfx event type string, e.g. ``"pointer_down"``.

        Returns
        -------
        int
            Number of times the event type was observed (0 if never).
        """
        return self._counts.get(event_type, 0)

    def total(self):
        """Return total number of events captured across all types.

        Returns
        -------
        int
            Sum of all event counts.
        """
        return sum(self._counts.values())

    def counts(self):
        """Return a copy of the full event_type to count mapping.

        Returns
        -------
        dict
            Copy of the internal counts dict.
        """
        return dict(self._counts)

    def clear(self):
        """Reset both the event log and all counters."""
        super().clear()
        self._counts.clear()

    def _on_event(self, event):
        """Increment count and delegate to parent recorder.

        Parameters
        ----------
        event : pygfx Event
            A live pygfx event object.
        """
        et = str(getattr(event, "type", "") or "unknown")
        self._counts[et] = self._counts.get(et, 0) + 1
        super()._on_event(event)


class EventPlayer:
    """Replays a sequence of RecordedEvent objects into a pygfx actor.

    Injects synthetic pygfx events via ``handle_event``, making them
    indistinguishable from real user input.

    Parameters
    ----------
    recorder : EventRecorder, optional
        Source of events to replay.
    speed_factor : float, optional
        Multiplier applied to inter-event delays.
        ``1.0`` is real-time; ``0.0`` is instant (best for tests).

    Examples
    --------
    >>> player = EventPlayer(speed_factor=0.0)
    >>> player.speed_factor
    0.0
    """

    def __init__(self, recorder=None, speed_factor=1.0):
        """Initialise EventPlayer.

        Parameters
        ----------
        recorder : EventRecorder, optional
            Source of events to replay.
        speed_factor : float, optional
            Playback speed multiplier. ``0.0`` for instant replay.
        """
        self._recorder = recorder
        self.speed_factor = speed_factor

    def load(self, filepath):
        """Load events from a JSON file written by EventRecorder.save.

        Parameters
        ----------
        filepath : str
            Path to the saved session JSON file.
        """
        rec = EventRecorder()
        rec.load(filepath)
        self._recorder = rec

    def play(self, actor):
        """Replay all events into actor via handle_event.

        Parameters
        ----------
        actor : pygfx WorldObject
            A pygfx actor whose ``handle_event`` will receive the events.

        Raises
        ------
        RuntimeError
            If no events are available to replay.
        """
        if self._recorder is None:
            raise RuntimeError(
                "No events to replay. Pass a recorder or call load() first."
            )

        events = self._recorder.events
        if not events:
            return

        prev_ts = None
        for evt in events:
            if self.speed_factor > 0 and prev_ts is not None:
                delay = (evt.timestamp - prev_ts) / self.speed_factor
                if delay > 0:
                    time.sleep(delay)
            prev_ts = evt.timestamp
            actor.handle_event(evt.to_pygfx_event())
