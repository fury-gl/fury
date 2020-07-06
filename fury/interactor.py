"""Custom Interactor Style."""

from collections import deque

import numpy as np
import vtk


class Event(object):
    """Event class."""

    def __init__(self):
        self.position = None
        self.name = None
        self.key = None
        self.alt_key = None
        self.shift_key = None
        self.ctrl_key = None
        self._abort_flag = None

    @property
    def abort_flag(self):
        """Abort."""
        return self._abort_flag

    def update(self, event_name, interactor):
        """Update current event information."""
        self.name = event_name
        self.position = np.asarray(interactor.GetEventPosition())
        self.key = interactor.GetKeySym()
        self.alt_key = bool(interactor.GetAltKey())
        self.shift_key = bool(interactor.GetShiftKey())
        self.ctrl_key = bool(interactor.GetControlKey())
        self._abort_flag = False  # Reset abort flag

    def abort(self):
        """Abort the event i.e. do not propagate it any further."""
        self._abort_flag = True

    def reset(self):
        """Done with the current event. Reset the attributes."""
        self.position = None
        self.name = None
        self.key = None
        self._abort_flag = False


class CustomInteractorStyle(vtk.vtkInteractorStyleUser):
    """Manipulate the camera and interact with objects in the scene.

    This interactor style allows the user to interactively manipulate (pan,
    rotate and zoom) the camera. It also allows the user to interact (click,
    scroll, etc.) with objects in the scene.

    Several events handling methods from :class:`vtkInteractorStyleUser` have
    been overloaded to allow the propagation of the events to the objects the
    user is interacting with.

    In summary, while interacting with the scene, the mouse events are as
    follows::

        - Left mouse button: rotates the camera
        - Right mouse button: dollys the camera
        - Mouse wheel: dollys the camera
        - Middle mouse button: pans the camera

    """

    def __init__(self):
        """Init."""
        # Interactor responsible for moving the camera.
        self.trackball_camera = vtk.vtkInteractorStyleTrackballCamera()
        # Interactor responsible for moving/rotating a selected actor.
        self.trackball_actor = vtk.vtkInteractorStyleTrackballActor()
        # Interactor responsible for panning/zooming the camera.
        self.image = vtk.vtkInteractorStyleImage()

        # The picker allows us to know which object/actor is under the mouse.
        self.picker = vtk.vtkPropPicker()
        self.chosen_element = None
        self.event = Event()
        self.event2id = {}  # To map an event's name to an ID.

        # Define some interaction states
        self.left_button_down = False
        self.right_button_down = False
        self.middle_button_down = False
        self.active_props = set()

        self.history = deque(maxlen=10)  # Events history.

        self.selected_props = {"left_button": set(),
                               "right_button": set(),
                               "middle_button": set()}

    def add_active_prop(self, prop):
        self.active_props.add(prop)

    def remove_active_prop(self, prop):
        self.active_props.remove(prop)

    def get_prop_at_event_position(self):
        """Return the prop that lays at the event position."""
        # TODO: return a list of items (i.e. each level of the assembly path).
        event_pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(event_pos[0], event_pos[1], 0,
                         self.GetCurrentRenderer())

        path = self.picker.GetPath()
        if path is None:
            return None

        node = path.GetLastNode()
        prop = node.GetViewProp()
        return prop

    def propagate_event(self, evt, *props):
        evt = self.event2id.get(evt, evt)
        for prop in props:
            # Propagate event to the prop.
            if prop is not None:
                prop.InvokeEvent(evt)

            if self.event.abort_flag:
                return

    def _process_event(self, obj, evt):
        self.event.update(evt, self.GetInteractor())
        self.history.append({
            "event": evt,
            "pos": self.event.position,
        })

        if evt == "LeftButtonPressEvent":
            self.on_left_button_down(obj, evt)
        elif evt == "LeftButtonReleaseEvent":
            self.on_left_button_up(obj, evt)
        elif evt == "RightButtonPressEvent":
            self.on_right_button_down(obj, evt)
        elif evt == "RightButtonReleaseEvent":
            self.on_right_button_up(obj, evt)
        elif evt == "MiddleButtonPressEvent":
            self.on_middle_button_down(obj, evt)
        elif evt == "MiddleButtonReleaseEvent":
            self.on_middle_button_up(obj, evt)
        elif evt == "MouseMoveEvent":
            self.on_mouse_move(obj, evt)
        elif evt == "CharEvent":
            self.on_char(obj, evt)
        elif evt == "KeyPressEvent":
            self.on_key_press(obj, evt)
        elif evt == "KeyReleaseEvent":
            self.on_key_release(obj, evt)
        elif evt == "MouseWheelForwardEvent":
            self.on_mouse_wheel_forward(obj, evt)
        elif evt == "MouseWheelBackwardEvent":
            self.on_mouse_wheel_backward(obj, evt)

        self.event.reset()  # Event fully processed.

    def _button_clicked(self, button, last_event=-1, before_last_event=-2):
        if len(self.history) < abs(before_last_event):
            return False

        if self.history[last_event]["event"] != button + "ButtonReleaseEvent":
            return False

        if self.history[before_last_event]["event"] \
                != button + "ButtonPressEvent":
            return False

        return True

    def _button_double_clicked(self, button):
        if not (self._button_clicked(button) and
                self._button_clicked(button, -3, -4)):
            return False

        return True

    def on_left_button_down(self, _obj, evt):
        self.left_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["left_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.trackball_camera.OnLeftButtonDown()

    def on_left_button_up(self, _obj, evt):
        self.left_button_down = False
        self.propagate_event(evt, *self.selected_props["left_button"])
        self.selected_props["left_button"].clear()
        self.trackball_camera.OnLeftButtonUp()

        prop = self.get_prop_at_event_position()

        if self._button_double_clicked("Left"):
            self.propagate_event("LeftButtonDoubleClickEvent", prop)
            self.history.clear()

    def on_right_button_down(self, _obj, evt):
        self.right_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["right_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.trackball_camera.OnRightButtonDown()

    def on_right_button_up(self, _obj, evt):
        self.right_button_down = False
        self.propagate_event(evt, *self.selected_props["right_button"])
        self.selected_props["right_button"].clear()
        self.trackball_camera.OnRightButtonUp()

        if self._button_double_clicked("Right"):
            prop = self.get_prop_at_event_position()
            self.propagate_event("RightButtonDoubleClickEvent", prop)
            self.history.clear()

    def on_middle_button_down(self, _obj, evt):
        self.middle_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["middle_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.trackball_camera.OnMiddleButtonDown()

    def on_middle_button_up(self, _obj, evt):
        self.middle_button_down = False
        self.propagate_event(evt, *self.selected_props["middle_button"])
        self.selected_props["middle_button"].clear()
        self.trackball_camera.OnMiddleButtonUp()

        if self._button_double_clicked("Middle"):
            prop = self.get_prop_at_event_position()
            self.propagate_event("MiddleButtonDoubleClickEvent", prop)
            self.history.clear()

    def on_mouse_move(self, _obj, evt):
        """On mouse move."""
        # Only propagate events to active or selected props.
        self.propagate_event(evt, *(self.active_props |
                                    self.selected_props["left_button"] |
                                    self.selected_props["right_button"] |
                                    self.selected_props["middle_button"]))

        self.trackball_camera.OnMouseMove()

    def on_mouse_wheel_forward(self, _obj, evt):
        """On mouse wheel forward."""
        # First, propagate mouse wheel event to underneath prop.
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.propagate_event(evt, prop)

        # Then, to the active props.
        if not self.event.abort_flag:
            self.propagate_event(evt, *self.active_props)

        # Finally, to the default interactor.
        if not self.event.abort_flag:
            self.trackball_camera.OnMouseWheelForward()

    def on_mouse_wheel_backward(self, _obj, evt):
        """On mouse wheel backward."""
        # First, propagate mouse wheel event to underneath prop.
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.propagate_event(evt, prop)

        # Then, to the active props.
        if not self.event.abort_flag:
            self.propagate_event(evt, *self.active_props)

        # Finally, to the default interactor.
        if not self.event.abort_flag:
            self.trackball_camera.OnMouseWheelBackward()

    def on_char(self, _obj, evt):
        self.propagate_event(evt, *self.active_props)

    def on_key_press(self, _obj, evt):
        self.propagate_event(evt, *self.active_props)

    def on_key_release(self, _obj, evt):
        self.propagate_event(evt, *self.active_props)

    def SetInteractor(self, interactor):
        """Define new interactor."""
        # Internally, `InteractorStyle` objects need a handle to a
        # `vtkWindowInteractor` object and this is done via `SetInteractor`.
        # However, this has the side effect of adding directly all their
        # observers to the `interactor`!
        self.trackball_actor.SetInteractor(interactor)
        self.image.SetInteractor(interactor)
        self.trackball_camera.SetInteractor(interactor)

        # Remove all observers *most likely* (cannot guarantee that the
        # interactor did not already have these observers) added by
        # `vtkInteractorStyleTrackballCamera`, i.e. our `trackball_camera`.
        #
        # Note: Be sure that no observer has been manually added to the
        # `interactor` before setting the InteractorStyle.
        interactor.RemoveObservers("TimerEvent")
        interactor.RemoveObservers("EnterEvent")
        interactor.RemoveObservers("LeaveEvent")
        interactor.RemoveObservers("ExposeEvent")
        interactor.RemoveObservers("ConfigureEvent")
        interactor.RemoveObservers("CharEvent")
        interactor.RemoveObservers("KeyPressEvent")
        interactor.RemoveObservers("KeyReleaseEvent")
        interactor.RemoveObservers("MouseMoveEvent")
        interactor.RemoveObservers("LeftButtonPressEvent")
        interactor.RemoveObservers("RightButtonPressEvent")
        interactor.RemoveObservers("MiddleButtonPressEvent")
        interactor.RemoveObservers("LeftButtonReleaseEvent")
        interactor.RemoveObservers("RightButtonReleaseEvent")
        interactor.RemoveObservers("MiddleButtonReleaseEvent")
        interactor.RemoveObservers("MouseWheelForwardEvent")
        interactor.RemoveObservers("MouseWheelBackwardEvent")

        # This class is a `vtkClass` (instead of `object`), so `super()`
        # cannot be used. Also the method `SetInteractor` is not overridden in
        # `vtkInteractorStyleUser` so we have to call directly the one from
        # `vtkInteractorStyle`. In addition to setting the interactor, the
        # following line adds the necessary hooks to listen to this
        # observers instances.
        vtk.vtkInteractorStyle.SetInteractor(self, interactor)

        # Keyboard events.
        self.AddObserver("CharEvent", self._process_event)
        self.AddObserver("KeyPressEvent", self._process_event)
        self.AddObserver("KeyReleaseEvent", self._process_event)

        # Mouse events.
        self.AddObserver("MouseMoveEvent", self._process_event)
        self.AddObserver("LeftButtonPressEvent", self._process_event)
        self.AddObserver("LeftButtonReleaseEvent", self._process_event)
        self.AddObserver("RightButtonPressEvent", self._process_event)
        self.AddObserver("RightButtonReleaseEvent", self._process_event)
        self.AddObserver("MiddleButtonPressEvent", self._process_event)
        self.AddObserver("MiddleButtonReleaseEvent", self._process_event)

        # Windows and special events.
        # TODO: we ever find them useful we could support them.
        # self.AddObserver("TimerEvent", self._process_event)
        # self.AddObserver("EnterEvent", self._process_event)
        # self.AddObserver("LeaveEvent", self._process_event)
        # self.AddObserver("ExposeEvent", self._process_event)
        # self.AddObserver("ConfigureEvent", self._process_event)

        # These observers need to be added directly to the interactor because
        # `vtkInteractorStyleUser` does not support wheel events prior 7.1. See
        # https://github.com/Kitware/VTK/commit/373258ed21f0915c425eddb996ce6ac13404be28
        interactor.AddObserver("MouseWheelForwardEvent", self._process_event)
        interactor.AddObserver("MouseWheelBackwardEvent", self._process_event)

    def force_render(self):
        """Causes the scene to refresh."""
        self.GetInteractor().GetRenderWindow().Render()

    def add_callback(self, prop, event_type, callback, priority=0, args=[]):
        """Add a callback associated to a specific event for a VTK prop.

        Parameters
        ----------
        prop : vtkProp
        event_type : event code
        callback : function
        priority : int
        """

        def _callback(_obj, event_name):
            # Update event information.
            interactor_ = self.GetInteractor()
            if interactor_ is not None:
                callback(self, prop, *args)
            else:
                print('interactor is none')
                print('event name is', event_name)

        # Dealing with custom events not defined in VTK.
        # Check whether the Event is predefined or not.
        if vtk.vtkCommand.GetEventIdFromString(event_type) == 0:
            if event_type not in self.event2id:
                # If the event type was not previously defined,
                # then create an extra user defined event.
                self.event2id[event_type] = \
                    vtk.vtkCommand.UserEvent + len(self.event2id) + 1

            event_type = self.event2id[event_type]

        prop.AddObserver(event_type, _callback, priority)
