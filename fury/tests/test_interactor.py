from os.path import join as pjoin
from collections import defaultdict

import numpy as np
import vtk

from fury import actor, window, interactor
from fury import utils as vtk_utils
from fury.data import DATA_DIR
from fury.decorators import skip_osx, skip_win

import numpy.testing as npt
import pytest


@pytest.mark.skipif(skip_osx or skip_win, reason="This test does not work on"
                                                 " Windows and OSX. Need to "
                                                 " be introspected")
def test_custom_interactor_style_events(recording=False):
    print("Using VTK {}".format(vtk.vtkVersion.GetVTKVersion()))
    filename = "test_custom_interactor_style_events.log.gz"
    recording_filename = pjoin(DATA_DIR, filename)
    scene = window.Scene()

    # the show manager allows to break the rendering process
    # in steps so that the widgets can be added properly
    interactor_style = interactor.CustomInteractorStyle()
    show_manager = window.ShowManager(scene, size=(800, 800),
                                      reset_camera=False,
                                      interactor_style=interactor_style)

    # Create a cursor, a circle that will follow the mouse.
    polygon_source = vtk.vtkRegularPolygonSource()
    polygon_source.GeneratePolygonOff()  # Only the outline of the circle.
    polygon_source.SetNumberOfSides(50)
    polygon_source.SetRadius(10)
    # polygon_source.SetRadius
    polygon_source.SetCenter(0, 0, 0)

    mapper = vtk.vtkPolyDataMapper2D()
    vtk_utils.set_input(mapper, polygon_source.GetOutputPort())

    cursor = vtk.vtkActor2D()
    cursor.SetMapper(mapper)
    cursor.GetProperty().SetColor(1, 0.5, 0)
    scene.add(cursor)

    def follow_mouse(iren, obj):
        obj.SetPosition(*iren.event.position)
        iren.force_render()

    interactor_style.add_active_prop(cursor)
    interactor_style.add_callback(cursor, "MouseMoveEvent", follow_mouse)

    # create some minimalistic streamlines
    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0.3, 0.7, 0.]])
    tube1 = actor.streamtube([lines[0]], colors[0])
    tube2 = actor.streamtube([lines[1]], colors[1])
    scene.add(tube1)
    scene.add(tube2)

    # Define some counter callback.
    states = defaultdict(lambda: 0)

    def counter(iren, _obj):
        states[iren.event.name] += 1

    # Assign the counter callback to every possible event.
    for event in ["CharEvent", "MouseMoveEvent",
                  "KeyPressEvent", "KeyReleaseEvent",
                  "LeftButtonPressEvent", "LeftButtonReleaseEvent",
                  "RightButtonPressEvent", "RightButtonReleaseEvent",
                  "MiddleButtonPressEvent", "MiddleButtonReleaseEvent"]:
        interactor_style.add_callback(tube1, event, counter)

    # Add callback to scale up/down tube1.
    def scale_up_obj(iren, obj):
        counter(iren, obj)
        scale = np.asarray(obj.GetScale()) + 0.1
        obj.SetScale(*scale)
        iren.force_render()
        iren.event.abort()  # Stop propagating the event.

    def scale_down_obj(iren, obj):
        counter(iren, obj)
        scale = np.array(obj.GetScale()) - 0.1
        obj.SetScale(*scale)
        iren.force_render()
        iren.event.abort()  # Stop propagating the event.

    interactor_style.add_callback(tube2, "MouseWheelForwardEvent",
                                  scale_up_obj)
    interactor_style.add_callback(tube2, "MouseWheelBackwardEvent",
                                  scale_down_obj)

    # Add callback to hide/show tube1.
    def toggle_visibility(iren, obj):
        key = iren.event.key
        if key.lower() == "v":
            obj.SetVisibility(not obj.GetVisibility())
            iren.force_render()

    interactor_style.add_active_prop(tube1)
    interactor_style.add_active_prop(tube2)
    interactor_style.remove_active_prop(tube2)
    interactor_style.add_callback(tube1, "CharEvent", toggle_visibility)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(states.items()))
    else:
        show_manager.play_events_from_file(recording_filename)
        msg = ("Wrong count for '{}'.")
        expected = [('CharEvent', 6),
                    ('KeyPressEvent', 6),
                    ('KeyReleaseEvent', 6),
                    ('MouseMoveEvent', 1652),
                    ('LeftButtonPressEvent', 1),
                    ('RightButtonPressEvent', 1),
                    ('MiddleButtonPressEvent', 2),
                    ('LeftButtonReleaseEvent', 1),
                    ('MouseWheelForwardEvent', 3),
                    ('MouseWheelBackwardEvent', 1),
                    ('MiddleButtonReleaseEvent', 2),
                    ('RightButtonReleaseEvent', 1)]

        # Useful loop for debugging.
        for event, count in expected:
            if states[event] != count:
                print("{}: {} vs. {} (expected)".format(event,
                                                        states[event],
                                                        count))

        for event, count in expected:
            npt.assert_equal(states[event], count, err_msg=msg.format(event))


if __name__ == '__main__':
    test_custom_interactor_style_events(recording=True)
