"""Utilities for testing."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import partial
import io
import json
import operator
import os
import sys
import time
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import scipy  # type: ignore

from fury.window import ShowManager


@contextmanager
def captured_output():
    """
    Capture stdout and stderr from print or logging.

    This context manager temporarily replaces sys.stdout and sys.stderr
    to capture printed output and return it for testing.

    Returns
    -------
    out : StringIO
        Object containing captured stdout.
    err : StringIO
        Object containing captured stderr.

    Examples
    --------
    >>> def foo():
    ...    print('hello world!')
    >>> with captured_output() as (out, err):
    ...    foo()
    >>> print(out.getvalue().strip())
    hello world!
    """
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def snapshot_report(scene, *, colors=None):
    """
    Render ``scene`` offscreen and analyze the resulting image.

    A thin wrapper around :func:`fury.window.snapshot` (with
    ``return_array=True``, writing no file) and
    :func:`fury.window.analyze_snapshot`. Performs a real offscreen render --
    no mocks, patches, or dummy objects.

    Parameters
    ----------
    scene : Scene
        The scene to render.
    colors : sequence of (r, g, b), optional
        Expected colors in ``[0, 1]``. When given, the returned report's
        ``colors_found`` reports whether each color is present in the image.

    Returns
    -------
    ReportSnapshot
        The analysis report, exposing ``objects`` (count of foreground blobs)
        and ``colors_found``.
    """
    from fury.colormap import colors_to_uint8
    from fury.window import analyze_snapshot, snapshot

    rgb255 = colors_to_uint8(colors) if colors is not None else None
    arr = snapshot(scene=scene, fname=None, return_array=True)
    return analyze_snapshot(arr, colors=rgb255, find_objects=True)


def assert_visibility(target, *, toggle, colors=None):
    """
    Assert ``target`` renders when visible and renders nothing when hidden.

    Drives a real offscreen render through :func:`snapshot_report` and uses the
    background-difference object count from
    :func:`fury.window.analyze_snapshot` as the visibility signal. No mocks,
    patches, or dummy classes are involved -- visibility is toggled through the
    object's real API via ``toggle``.

    Parameters
    ----------
    target : Object
        An actor or UI element. It is added to a fresh
        :class:`~fury.window.Scene` for each render (``window.snapshot`` attaches
        a camera to the scene, so a new scene is used per snapshot).
    toggle : callable
        The object's real visibility setter, invoked as ``toggle(True)`` and
        ``toggle(False)`` (e.g. ``element.set_visibility`` for UI, or
        ``lambda v: setattr(actor, "visible", v)`` for a raw actor).
    colors : sequence of (r, g, b), optional
        Expected colors in ``[0, 1]``. When given, they must be present while
        visible and absent while hidden. Reliable only for flat/``"basic"``
        materials where the rendered color matches exactly.
    """
    from fury.window import Scene

    def _render():
        """
        Render function to check visibility.

        Returns
        -------
        ReportSnapshot
            The analysis report, exposing ``objects`` (count of foreground blobs)
            and ``colors_found``.
        """
        scene = Scene()
        scene.add(target)
        return snapshot_report(scene, colors=colors)

    toggle(True)
    report = _render()
    assert report.objects >= 1, "expected a visible object to render"
    if colors is not None:
        assert all(report.colors_found), "expected colors missing while visible"

    toggle(False)
    report = _render()
    assert report.objects == 0, "expected nothing to render while hidden"
    if colors is not None:
        assert not any(report.colors_found), "colors present while hidden"

    toggle(True)


def assert_operator(value1, value2, *, msg="", op=operator.eq):
    """
    Check boolean statement using the given operator.

    Compares two values using the provided operator and raises
    an AssertionError if the comparison is false.

    Parameters
    ----------
    value1 : object
        First value to be compared.
    value2 : object
        Second value to be compared.
    msg : str, optional
        Error message to be displayed if the assertion fails.
        Can contain format placeholders for values.
    op : callable, optional
        Operator to compare values. Default is equality operator.

    Raises
    ------
    AssertionError
        If the comparison between value1 and value2 using op returns False.
    """
    if not op(value1, value2):
        raise AssertionError(msg.format(str(value2), str(value1)))


assert_greater_equal = partial(
    assert_operator,
    op=operator.ge,
    msg="{0} >= {1}",
)
assert_greater = partial(assert_operator, op=operator.gt, msg="{0} > {1}")
assert_less_equal = partial(assert_operator, op=operator.le, msg="{0} =< {1}")
assert_less = partial(assert_operator, op=operator.lt, msg="{0} < {1}")
assert_true = partial(
    assert_operator, value2=True, op=operator.eq, msg="False is not true"
)
assert_false = partial(
    assert_operator, value2=False, op=operator.eq, msg="True is not false"
)
assert_not_equal = partial(assert_operator, op=operator.ne)
assert_equal = partial(assert_operator, op=operator.eq)


def assert_arrays_equal(arrays1, arrays2):
    """
    Check that all arrays in arrays1 equal the corresponding arrays in arrays2.

    Parameters
    ----------
    arrays1 : sequence of ndarray
        First sequence of arrays to be compared.
    arrays2 : sequence of ndarray
        Second sequence of arrays to be compared.

    Raises
    ------
    AssertionError
        If any corresponding arrays are not equal.
    """
    for arr1, arr2 in zip(arrays1, arrays2, strict=False):
        assert_array_equal(arr1, arr2)


class VisualTest:
    """
    Visual testing harness for UI components.

    Parameters
    ----------
    events_names : list of str, optional
        Event names to track. If None, a default set of pointer/key events
        is used.
    """

    def __init__(self, *, events_names=None):
        """Initialize the visual test harness."""
        if events_names is None:
            events_names = [
                "pointer_down",
                "pointer_up",
                "pointer_move",
                "pointer_drag",
                "key_down",
                "key_up",
                "click",
                "double_click",
                "wheel",
            ]

        self.events_counts = dict.fromkeys(events_names, 0)
        self._ui_snapshots = {}
        self._ui_components = {}

    def count(self, event):
        """
        Increment the count of a registered event target.

        Parameters
        ----------
        event : object
            The target event being triggered.
        """
        target = getattr(event, "target", getattr(event, "_target", None))
        current_target = getattr(
            event, "current_target", getattr(event, "_current_target", None)
        )
        if target == current_target:
            self.events_counts[event.type] += 1

    def monitor(self, ui_component, name=None):
        """
        Register callbacks to monitor events on a UI component.

        Also registers the component for UI property snapshotting.

        Parameters
        ----------
        ui_component : object
            UI component with actors to monitor for events.
        name : str, optional
            Identifier for this component in snapshots. If None, the
            class name is used.
        """
        for event in self.events_counts:
            for obj_actor in get_all_actors(ui_component):
                ui_component.add_callback(obj_actor, event, self.count)

        if name is None:
            name = ui_component.__class__.__name__
        self._ui_components[name] = ui_component

    def snapshot_ui(self, name, ui_component, step=0):
        """
        Capture current visual properties of a UI component.

        Records position and size. Called automatically before and after
        event playback to detect visual regressions.

        Parameters
        ----------
        name : str
            Identifier for this UI component snapshot.
        ui_component : object
            The UI component to snapshot. Must have ``get_position()``
            and ``size`` attributes.
        step : int, optional
            The current step/event index of the snapshot (default 0).
        """
        snapshot = {}
        if hasattr(ui_component, "get_position"):
            snapshot["position"] = tuple(float(v) for v in ui_component.get_position())
        if hasattr(ui_component, "size"):
            snapshot["size"] = tuple(int(v) for v in ui_component.size)

        step_str = str(step)
        if step_str not in self._ui_snapshots:
            self._ui_snapshots[step_str] = {}
        self._ui_snapshots[step_str][name] = snapshot

    def snapshot_all_ui(self, step=0):
        """
        Capture visual properties of all monitored UI components at a step.

        Parameters
        ----------
        step : int, optional
            The step/event index of the snapshot (default 0).
        """
        for name, comp in self._ui_components.items():
            self.snapshot_ui(name, comp, step=step)

    def snapshot_window(self, show_manager):
        """
        Capture current window configuration from the ShowManager.

        Parameters
        ----------
        show_manager : ShowManager
            The ShowManager instance to snapshot.

        Returns
        -------
        dict
            Window configuration snapshot with keys ``'size'`` and
            ``'screen_config'``.
        """
        return {
            "size": tuple(show_manager._size),
            "screen_config": show_manager._screen_config,
        }

    def validate_ui(self, step, name, expected):
        """
        Assert UI component properties match expected values.

        Parameters
        ----------
        step : int or str
            The step/event index of the snapshot to validate.
        name : str
            Identifier of the previously snapshotted UI component.
        expected : dict
            Expected property values. Keys can include ``'position'``
            and ``'size'``.

        Raises
        ------
        AssertionError
            If no snapshot exists for ``name`` at ``step`` or if property mismatches.
        """
        step_str = str(step)
        step_snapshots = self._ui_snapshots.get(step_str)
        if step_snapshots is None:
            raise AssertionError(f"No snapshots found for step '{step}'")
        actual = step_snapshots.get(name)
        if actual is None:
            raise AssertionError(f"No snapshot found for '{name}' at step '{step}'")
        for key, expected_val in expected.items():
            act = actual.get(key)
            if isinstance(act, (list, tuple)):
                act = list(act)
            if isinstance(expected_val, (list, tuple)):
                expected_val = list(expected_val)
            assert_equal(
                act,
                expected_val,
                msg=f"UI property '{key}' mismatch for '{name}' at step '{step}'",
            )

    def validate_window(self, show_manager, expected):
        """
        Assert window configuration matches expected values.

        Parameters
        ----------
        show_manager : ShowManager
            The ShowManager instance to validate.
        expected : dict
            Expected values. Keys can include ``'size'`` and
            ``'screen_config'``.

        Raises
        ------
        AssertionError
            If any window property mismatches.
        """
        actual = self.snapshot_window(show_manager)
        for key, expected_val in expected.items():
            act = actual.get(key)
            if isinstance(act, (list, tuple)):
                act = list(act)
            if isinstance(expected_val, (list, tuple)):
                expected_val = list(expected_val)
            assert_equal(
                act,
                expected_val,
                msg=f"Window property '{key}' mismatch",
            )

    def record_or_test(
        self,
        recording_filename,
        expected_filename,
        size=(600, 600),
        title="FURY VisualTest",
        screen_config=None,
        recording=False,
    ):
        """
        Record interactions if recording is True, otherwise simulate or test.

        If the environment variable FURY_SHOW_SIMULATION is set, show the
        simulation and close the window. Otherwise, validate the event
        counts and UI snapshots.

        Parameters
        ----------
        recording_filename : str
            Path to recorded events file.
        expected_filename : str
            Path to save or load expected counts/snapshots JSON.
        size : tuple of int, optional
            Window size (width, height). Default (600, 600).
        title : str, optional
            Window title. Default 'FURY VisualTest'.
        screen_config : list, optional
            Screen configuration for the ShowManager.
        recording : bool, optional
            Whether to record the events (default False).
        """

        show_simulation = os.environ.get("FURY_SHOW_SIMULATION", "").lower() in (
            "true",
            "1",
        )
        w_type = "glfw" if (recording or show_simulation) else "offscreen"

        show_manager = ShowManager(
            size=size,
            title=title,
            window_type=w_type,
            screen_config=screen_config,
        )

        for ui_comp in self._ui_components.values():
            show_manager.scene.add(ui_comp)

        # Snapshot step 0 (startup)
        self.snapshot_all_ui(step=0)

        if recording:
            record_events_to_file(show_manager, recording_filename, visual_test=self)
            # Final snapshot at the end of recording
            total_events = len(show_manager._recorded_events)
            self.snapshot_all_ui(step=total_events)
            self.save(expected_filename, show_manager)
        else:
            expected = self.load(expected_filename)
            expected_snapshots = expected._ui_snapshots

            play_events_from_file(
                show_manager,
                recording_filename,
                visual_test=self,
                expected_ui_snapshots=expected_snapshots,
                show_simulation=show_simulation,
            )

            # Re-snapshot final step if it was recorded
            step_keys = [int(k) for k in expected_snapshots.keys() if k.isdigit()]
            if step_keys:
                max_step = max(step_keys)
                if str(max_step) not in self._ui_snapshots:
                    self.snapshot_all_ui(step=max_step)

            if show_simulation:
                time.sleep(0.5)
                show_manager.close()
            else:
                self.check_counts(expected)
                for step_str, comp_snapshots in expected_snapshots.items():
                    for name, snapshot in comp_snapshots.items():
                        self.validate_ui(step_str, name, snapshot)

    def save(self, filename, show_manager=None):
        """
        Serialize event counts, UI snapshots, and window config to JSON.

        Optimizes file size by only saving visual properties that have changed
        from the previous step.

        Parameters
        ----------
        filename : str
            Path to save the JSON serialized data.
        show_manager : ShowManager, optional
            The ShowManager instance to get window config from.
        """
        filtered_snapshots = {}
        sorted_steps = sorted(int(k) for k in self._ui_snapshots.keys())
        last_state = {}

        for step in sorted_steps:
            step_str = str(step)
            step_data = self._ui_snapshots[step_str]
            filtered_step_data = {}
            for comp_name, comp_data in step_data.items():
                if comp_name not in last_state:
                    last_state[comp_name] = {}

                changed_properties = {}
                for prop, val in comp_data.items():
                    if (
                        prop not in last_state[comp_name]
                        or last_state[comp_name][prop] != val
                    ):
                        changed_properties[prop] = val
                        last_state[comp_name][prop] = val

                if changed_properties:
                    filtered_step_data[comp_name] = changed_properties

            if filtered_step_data or step_str == "0":
                filtered_snapshots[step_str] = filtered_step_data

        data = {
            "events_counts": self.events_counts,
            "ui_snapshots": filtered_snapshots,
            "window": self.snapshot_window(show_manager) if show_manager else {},
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """
        Load serialized event counts and UI snapshots from a JSON file.

        Reconstructs the full visual properties by forward-filling unchanged
        values from previous steps.

        Parameters
        ----------
        filename : str
            Path to the target input JSON file.

        Returns
        -------
        VisualTest
            A new instance with loaded state.
        """
        vt = cls()
        with open(filename) as f:
            data = json.load(f)

        vt.events_counts = data["events_counts"]
        ui_snapshots = data.get("ui_snapshots", {})

        is_nested = False
        if ui_snapshots:
            first_key = list(ui_snapshots.keys())[0]
            if first_key.isdigit():
                is_nested = True

        if is_nested:
            resolved_snapshots = {}
            sorted_steps = sorted(int(k) for k in ui_snapshots.keys())
            last_state = {}

            for step in sorted_steps:
                step_str = str(step)
                step_data = ui_snapshots[step_str]

                for comp_name, comp_data in step_data.items():
                    if comp_name not in last_state:
                        last_state[comp_name] = {}
                    for prop, val in comp_data.items():
                        last_state[comp_name][prop] = val

                resolved_snapshots[step_str] = {
                    cname: dict(cdata) for cname, cdata in last_state.items()
                }
            vt._ui_snapshots = resolved_snapshots
        else:
            vt._ui_snapshots = {"0": ui_snapshots}

        return vt

    def check_counts(self, expected):
        """
        Compare recorded event counts against expected values.

        Parameters
        ----------
        expected : VisualTest
            VisualTest instance representing reference values.
        """
        assert_equal(len(self.events_counts), len(expected.events_counts))

        msg = "{}: {} vs. {} (expected)"
        for event, count in expected.events_counts.items():
            if self.events_counts[event] != count:
                print(msg.format(event, self.events_counts[event], count))

        msg = "Wrong count for '{}'."
        for event, count in expected.events_counts.items():
            assert_equal(
                self.events_counts[event],
                count,
                msg=msg.format(event),
            )


class clear_and_catch_warnings(warnings.catch_warnings):
    """
    Context manager that resets warning registry for catching warnings.

    Warnings can be slippery, because whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module. This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters. This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. Default is True.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit.

    Notes
    -----
    This class is copied (with minor modifications) from the Nibabel package.
    https://github.com/nipy/nibabel. See COPYING file distributed along with
    the Nibabel package for the copyright and license terms.

    Examples
    --------
    >>> import warnings
    >>> with clear_and_catch_warnings(modules=[np.random.rand]):
    ...     warnings.simplefilter('always')
    ...     # do something that raises a warning in np.random.rand
    """

    class_modules = ()

    def __init__(self, *, record=True, modules=()):
        """Initialize the context manager."""
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(clear_and_catch_warnings, self).__init__(record=record)

    def __enter__(self):
        """
        Clear warning registry for given modules.

        Returns
        -------
        clear_and_catch_warnings
            The context manager instance.
        """
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(clear_and_catch_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        """
        Restore warning registry to its previous state.

        Parameters
        ----------
        *exc_info : tuple
            Exception information, if any, raised in the context.
        """
        super(clear_and_catch_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


def setup_test():
    """
    Set numpy print options to "legacy" for new versions of numpy.

    Configure numpy print options to maintain compatibility with older versions.
    If imported into a file, nosetest will run this before any doctests.

    References
    ----------
    https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
    https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
    https://github.com/nipy/nibabel/pull/556
    """
    if LooseVersion(np.__version__) >= LooseVersion("1.14"):
        np.set_printoptions(legacy="1.13")

    # Temporary fix until scipy release in October 2018
    # must be removed after that
    # print the first occurrence of matching warnings for each location
    # (module + line number) where the warning is issued
    if (
        LooseVersion(np.__version__) >= LooseVersion("1.15")
        and LooseVersion(scipy.version.short_version) <= "1.1.0"
    ):
        warnings.simplefilter("default")


def check_for_warnings(warn_printed, w_msg):
    """
    Check for specific warnings in the warning registry.

    Parameters
    ----------
    warn_printed : list
        List of captured warnings.
    w_msg : str
        Warning message to check for.
    """
    selected_w = [w for w in warn_printed if issubclass(w.category, UserWarning)]
    assert len(selected_w) >= 1
    msg = [str(m.message) for m in selected_w]
    assert_equal(w_msg in msg, True)


def get_all_actors(scene):
    """
    Recursively extract event-handling actors from a scene.

    Parameters
    ----------
    scene : Scene
        The root scene workspace or layout container.

    Returns
    -------
    list of object, shape (N,)
        Flat sequence of event-handling actor instances.
    """
    actors = []

    def traverse(obj):
        """
        Evaluate an object and descendants for event handlers.

        Parameters
        ----------
        obj : object
            The target scene graph node or layout.
        """
        if obj is None:
            return
        if obj in actors:
            return

        if hasattr(obj, "add_event_handler"):
            actors.append(obj)

        if hasattr(obj, "actors"):
            try:
                for act in obj.actors:
                    traverse(act)
            except Exception:
                pass

        if hasattr(obj, "children"):
            for child in obj.children:
                traverse(child)

        if hasattr(obj, "_children"):
            for child in obj._children:
                traverse(child)

    traverse(scene)
    if hasattr(scene, "main_scene"):
        traverse(scene.main_scene)
    if hasattr(scene, "ui_scene"):
        traverse(scene.ui_scene)

    return actors


def record_events(show_manager, visual_test=None):
    """
    Record user interaction events during a rendering session.

    Parameters
    ----------
    show_manager : ShowManager
        The pipeline coordinator dispatching scene events.
    visual_test : VisualTest, optional
        VisualTest instance to trigger intermediate checkpoints.
    """
    if getattr(show_manager, "_events_recording_active", False):
        return

    show_manager._events_recording_active = True
    show_manager._recorded_events = []
    show_manager._visual_test = visual_test
    show_manager._step_count = 0

    original_dispatch_event = show_manager.renderer.dispatch_event

    def wrapped_dispatch_event(event):
        """
        Intercept, log, and dispatch interaction event attributes.

        Parameters
        ----------
        event : Event
            The UI or window event instance.
        """
        if event.type in [
            "resize",
            "close",
            "focus",
            "blur",
            "pointer_enter",
            "pointer_leave",
        ]:
            original_dispatch_event(event)
            return

        all_actors = get_all_actors(show_manager.scene)
        target_index = -1
        if event.target in all_actors:
            target_index = all_actors.index(event.target)

        event_dict = {
            "class": event.__class__.__name__,
            "type": event.type,
            "target_index": target_index,
        }
        for attr in [
            "x",
            "y",
            "button",
            "buttons",
            "dy",
            "dx",
            "key",
            "modifiers",
            "ntouch",
            "clicks",
        ]:
            if hasattr(event, attr):
                event_dict[attr] = getattr(event, attr)

        show_manager._recorded_events.append(event_dict)

        # Snapshot periodically during recording
        show_manager._step_count += 1
        if show_manager._visual_test is not None:
            if show_manager._step_count % 5 == 0:
                show_manager._visual_test.snapshot_all_ui(step=show_manager._step_count)

        original_dispatch_event(event)

    show_manager.renderer.dispatch_event = wrapped_dispatch_event


def record_events_to_file(show_manager, filename, visual_test=None):
    """
    Serialize recorded interaction events to an external file.

    Parameters
    ----------
    show_manager : ShowManager
        The pipeline coordinator tracking active states.
    filename : str
        The target local filesystem output trace log path.
    visual_test : VisualTest, optional
        VisualTest instance to trigger intermediate checkpoints.
    """
    record_events(show_manager, visual_test=visual_test)
    show_manager.start()

    import gzip
    import json

    events_json = json.dumps(show_manager._recorded_events)
    if filename.endswith(".gz"):
        with gzip.open(filename, "wt", encoding="utf-8") as f:
            f.write(events_json)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(events_json)


def play_events(
    show_manager,
    events,
    visual_test=None,
    expected_ui_snapshots=None,
    show_simulation=False,
):
    """
    Simulate a pre-recorded sequence of event interactions.

    Parameters
    ----------
    show_manager : ShowManager
        The target rendering workspace actor receiving events.
    events : list of dict, shape (M,)
        The sequence of serialized historical trace log states.
    visual_test : VisualTest, optional
        VisualTest instance to trigger intermediate checkpoints.
    expected_ui_snapshots : dict, optional
        Loaded snapshots mapping step -> components -> properties to check.
    show_simulation : bool, optional
        Flag to enable or disable real-time canvas simulation.
    """
    show_manager.render()
    show_manager._draw_canvas()

    show_manager._playing_back = True
    step_count = 0
    try:
        from pygfx import Event, KeyboardEvent, PointerEvent

        all_actors = get_all_actors(show_manager.scene)

        for event_dict in events:
            cls_name = event_dict.get("class", "Event")
            evt_type = event_dict.get("type")
            target_index = event_dict.get("target_index", -1)

            kwargs = {}
            for attr in [
                "x",
                "y",
                "button",
                "buttons",
                "dy",
                "dx",
                "key",
                "modifiers",
                "ntouch",
                "clicks",
            ]:
                if attr in event_dict:
                    kwargs[attr] = event_dict[attr]

            if cls_name == "PointerEvent":
                event = PointerEvent(type=evt_type, **kwargs)
            elif cls_name == "KeyboardEvent":
                event = KeyboardEvent(type=evt_type, **kwargs)
            else:
                event = Event(type=evt_type, **kwargs)

            target = None
            if target_index != -1 and target_index < len(all_actors):
                target = all_actors[target_index]
            else:
                target = show_manager.renderer

            path = []
            curr = target
            while curr is not None:
                if curr not in path:
                    path.append(curr)
                curr = getattr(curr, "parent", None)

            if show_manager.scene not in path:
                path.append(show_manager.scene)
            if show_manager.renderer not in path:
                path.append(show_manager.renderer)

            step_count += 1
            if visual_test is not None and step_count % 5 == 0:
                visual_test.snapshot_all_ui(step=step_count)

            event._target = target
            for obj in path:
                if hasattr(obj, "handle_event"):
                    event._current_target = obj
                    obj.handle_event(event)
                    if getattr(event, "_propagation_stopped", False):
                        break

            if show_simulation:
                show_manager._draw_canvas()

                if not hasattr(show_manager.window, "draw"):
                    time.sleep(0.001)
                    try:
                        import glfw

                        glfw.poll_events()
                    except Exception:
                        pass

            if visual_test is not None and step_count == len(events):
                visual_test.snapshot_all_ui(step=step_count)
    finally:
        show_manager._playing_back = False


def play_events_from_file(
    show_manager,
    filename,
    visual_test=None,
    expected_ui_snapshots=None,
    show_simulation=False,
):
    """
    Deserialize and simulate events from a recorded trace file.

    Parameters
    ----------
    show_manager : ShowManager
        The target rendering workspace pipeline hosting simulated events.
    filename : str
        The local filesystem path mapping to serialized trace data.
    visual_test : VisualTest, optional
        VisualTest instance to trigger intermediate checkpoints.
    expected_ui_snapshots : dict, optional
        Loaded snapshots mapping step -> components -> properties to check.
    show_simulation : bool, optional
        Flag to enable or disable real-time canvas simulation.
    """
    import gzip
    import json

    if filename.endswith(".gz"):
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            events = json.loads(f.read())
    else:
        with open(filename, "r", encoding="utf-8") as f:
            events = json.loads(f.read())

    play_events(
        show_manager,
        events,
        visual_test=visual_test,
        expected_ui_snapshots=expected_ui_snapshots,
        show_simulation=show_simulation,
    )
