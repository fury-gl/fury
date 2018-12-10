"""Custom Interactor Style."""

import numpy as np
import vtk


class Event(object):
    """Event class."""

    def __init__(self):
        self.position = None
        self.name = None
        self.key = None
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

        self.trackball_interactor_style = vtk.vtkInteractorStyleTrackballActor()
        self.image_interactor_style = vtk.vtkInteractorStyleImage()

        # Default interactor is responsible for moving the camera.
        self.default_interactor = vtk.vtkInteractorStyleTrackballCamera()
        # The picker allows us to know which object/actor is under the mouse.
        self.picker = vtk.vtkPropPicker()
        self.chosen_element = None
        self.event = Event()

        # Define some interaction states
        self.left_button_down = False
        self.right_button_down = False
        self.middle_button_down = False
        self.active_props = set()

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
        for prop in props:
            # Propagate event to the prop.
            prop.InvokeEvent(evt)

            if self.event.abort_flag:
                return

    def _process_event(self, obj, evt):
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

    def on_left_button_down(self, obj, evt):
        self.left_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["left_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.default_interactor.OnLeftButtonDown()

    def on_left_button_up(self, obj, evt):
        self.left_button_down = False
        self.propagate_event(evt, *self.selected_props["left_button"])
        self.selected_props["left_button"].clear()
        self.default_interactor.OnLeftButtonUp()

    def on_right_button_down(self, obj, evt):
        self.right_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["right_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.default_interactor.OnRightButtonDown()

    def on_right_button_up(self, obj, evt):
        self.right_button_down = False
        self.propagate_event(evt, *self.selected_props["right_button"])
        self.selected_props["right_button"].clear()
        self.default_interactor.OnRightButtonUp()

    def on_middle_button_down(self, obj, evt):
        self.middle_button_down = True
        prop = self.get_prop_at_event_position()
        if prop is not None:
            self.selected_props["middle_button"].add(prop)
            self.propagate_event(evt, prop)

        if not self.event.abort_flag:
            self.default_interactor.OnMiddleButtonDown()

    def on_middle_button_up(self, obj, evt):
        self.middle_button_down = False
        self.propagate_event(evt, *self.selected_props["middle_button"])
        self.selected_props["middle_button"].clear()
        self.default_interactor.OnMiddleButtonUp()

    def on_mouse_move(self, obj, evt):
        """On mouse move."""
        # Only propagate events to active or selected props.
        self.propagate_event(evt, *(self.active_props |
                                    self.selected_props["left_button"] |
                                    self.selected_props["right_button"] |
                                    self.selected_props["middle_button"]))

        self.default_interactor.OnMouseMove()

    def on_mouse_wheel_forward(self, obj, evt):
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
            self.default_interactor.OnMouseWheelForward()

    def on_mouse_wheel_backward(self, obj, evt):
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
            self.default_interactor.OnMouseWheelBackward()

    def on_char(self, obj, evt):
        self.propagate_event(evt, *self.active_props)

    def on_key_press(self, obj, evt):
        self.propagate_event(evt, *self.active_props)

    def on_key_release(self, obj, evt):
        self.propagate_event(evt, *self.active_props)

    def SetInteractor(self, interactor):
        """Define new interactor."""
        # Internally, `InteractorStyle` objects need a handle to a
        # `vtkWindowInteractor` object and this is done via `SetInteractor`.
        # However, this has the side effect of adding directly all their
        # observers to the `interactor`!
        self.trackball_interactor_style.SetInteractor(interactor)
        self.image_interactor_style.SetInteractor(interactor)
        self.default_interactor.SetInteractor(interactor)

        # Remove all observers *most likely* (cannot guarantee that the
        # interactor did not already have these observers) added by
        # `vtkInteractorStyleTrackballCamera`, i.e. our `default_interactor`.
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
        def _callback(obj, event_name):
            # Update event information.
            self.event.update(event_name, self.GetInteractor())
            callback(self, prop, *args)

        prop.AddObserver(event_type, _callback, priority)


class InteractorStyleImageAndTrackballActor(vtk.vtkInteractorStyleUser):
    """ Interactive manipulation of the camera specialized for images that can
    also manipulates objects in the scene independent of each other.

    This interactor style allows the user to interactively manipulate (pan and
    zoom) the camera. It also allows the user to interact (rotate, pan, etc.)
    with objects in the scene independent of each other. It is specially
    designed to work with a grid of actors.

    Several events are overloaded from its superclass `vtkInteractorStyle`,
    hence the mouse bindings are different. (The bindings keep the camera's
    view plane normal perpendicular to the x-y plane.)

    In summary the mouse events for this interaction style are as follows:
    - Left mouse button: rotates the selected object around its center point
    - Ctrl + left mouse button: spins the selected object around its view plane normal
    - Shift + left mouse button: pans the selected object
    - Middle mouse button: pans the camera
    - Right mouse button: dollys the camera
    - Mouse wheel: dollys the camera

    """
    def __init__(self):
        self.trackball_interactor_style = vtk.vtkInteractorStyleTrackballActor()
        self.image_interactor_style = vtk.vtkInteractorStyleImage()

    def on_left_button_pressed(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonDown()

    def on_left_button_released(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonUp()

    def on_right_button_pressed(self, obj, evt):
        self.image_interactor_style.OnRightButtonDown()

    def on_right_button_released(self, obj, evt):
        self.image_interactor_style.OnRightButtonUp()

    def on_middle_button_pressed(self, obj, evt):
        self.image_interactor_style.OnMiddleButtonDown()

    def on_middle_button_released(self, obj, evt):
        self.image_interactor_style.OnMiddleButtonUp()

    def on_mouse_moved(self, obj, evt):
        self.trackball_interactor_style.OnMouseMove()
        self.image_interactor_style.OnMouseMove()

    def on_mouse_wheel_forward(self, obj, evt):
        self.image_interactor_style.OnMouseWheelForward()

    def on_mouse_wheel_backward(self, obj, evt):
        self.image_interactor_style.OnMouseWheelBackward()

    def SetInteractor(self, interactor):
        # Internally these `InteractorStyle` objects need an handle to a
        # `vtkWindowInteractor` object and this is done via `SetInteractor`.
        # However, this has a the side effect of adding directly their
        # observers to `interactor`!
        self.trackball_interactor_style.SetInteractor(interactor)
        self.image_interactor_style.SetInteractor(interactor)

        # Remove all observers previously set. Those were *most likely* set by
        # `vtkInteractorStyleTrackballActor` and `vtkInteractorStyleImage`.
        #
        # Note: Be sure that no observer has been manually added to the
        #       `interactor` before setting the InteractorStyle.
        interactor.RemoveAllObservers()

        # This class is a `vtkClass` (instead of `object`), so `super()` cannot be used.
        # Also the method `SetInteractor` is not overridden by `vtkInteractorStyleUser`
        # so we have to call directly the one from `vtkInteractorStyle`.
        # In addition to setting the interactor, the following line
        # adds the necessary hooks to listen to this instance's observers.
        vtk.vtkInteractorStyle.SetInteractor(self, interactor)

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_pressed)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_released)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_pressed)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_released)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_pressed)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_released)
        self.AddObserver("MouseMoveEvent", self.on_mouse_moved)

        # These observers need to be added directly to the interactor because
        # `vtkInteractorStyleUser` does not forward these events.
        interactor.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        interactor.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)


class InteractorStyleGrid(InteractorStyleImageAndTrackballActor):

    ANTICLOCKWISE_ROTATION_Y = np.array([-10, 0, 1, 0])
    CLOCKWISE_ROTATION_Y = np.array([10, 0, 1, 0])
    ANTICLOCKWISE_ROTATION_X = np.array([-10, 1, 0, 0])
    CLOCKWISE_ROTATION_X = np.array([10, 1, 0, 0])

    def __init__(self, bundles_actors):
        InteractorStyleImageAndTrackballActor.__init__(self)
        self.bundles_actors = bundles_actors

    def on_key_pressed(self, obj, evt):
        has_changed = False
        if obj.GetKeySym() == "Left":
            has_changed = True
            for a in self.bundles_actors:
                self.rotate(a, self.ANTICLOCKWISE_ROTATION_Y)
        elif obj.GetKeySym() == "Right":
            has_changed = True
            for a in self.bundles_actors:
                self.rotate(a, self.CLOCKWISE_ROTATION_Y)
        elif obj.GetKeySym() == "Up":
            has_changed = True
            for a in self.bundles_actors:
                self.rotate(a, self.ANTICLOCKWISE_ROTATION_X)
        elif obj.GetKeySym() == "Down":
            has_changed = True
            for a in self.bundles_actors:
                self.rotate(a, self.CLOCKWISE_ROTATION_X)

        if has_changed:
            obj.GetInteractor().Render()

    def SetInteractor(self, interactor):
        InteractorStyleImageAndTrackballActor.SetInteractor(self, interactor)
        self.AddObserver("KeyPressEvent", self.on_key_pressed)

    def rotate(self, prop3D, rotation):
        center = np.array(prop3D.GetCenter())

        oldMatrix = prop3D.GetMatrix()
        orig = np.array(prop3D.GetOrigin())

        newTransform = vtk.vtkTransform()
        newTransform.PostMultiply()
        if prop3D.GetUserMatrix() is not None:
            newTransform.SetMatrix(prop3D.GetUserMatrix())
        else:
            newTransform.SetMatrix(oldMatrix)

        newTransform.Translate(*(-center))
        newTransform.RotateWXYZ(*rotation)
        newTransform.Translate(*center)

        # now try to get the composit of translate, rotate, and scale
        newTransform.Translate(*(-orig))
        newTransform.PreMultiply()
        newTransform.Translate(*orig)

        if prop3D.GetUserMatrix() is not None:
            newTransform.GetMatrix(prop3D.GetUserMatrix())
        else:
            prop3D.SetPosition(newTransform.GetPosition())
            prop3D.SetScale(newTransform.GetScale())
            prop3D.SetOrientation(newTransform.GetOrientation())
