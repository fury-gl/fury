import os
import warnings
import numpy as np
from fury import actor, window
import numpy.testing as npt
from fury.decorators import xvfb_it

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
skip_it = use_xvfb == 'skip'


@npt.dec.skipif(skip_it)
@xvfb_it
def test_scene():

    scene = window.Scene()

    npt.assert_equal(scene.size(), (0, 0))

    # background color for scene (1, 0.5, 0)
    # 0.001 added here to remove numerical errors when moving from float
    # to int values
    bg_float = (1, 0.501, 0)

    # that will come in the image in the 0-255 uint scale
    bg_color = tuple((np.round(255 * np.array(bg_float))).astype('uint8'))

    scene.background(bg_float)
    # window.show(scene)
    arr = window.snapshot(scene)

    report = window.analyze_snapshot(arr,
                                     bg_color=bg_color,
                                     colors=[bg_color, (0, 127, 0)])
    npt.assert_equal(report.objects, 0)
    npt.assert_equal(report.colors_found, [True, False])

    axes = actor.axes()
    scene.add(axes)
    # window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    scene.rm(axes)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    window.add(scene, axes)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    scene.rm_all()
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    ren2 = window.Scene(bg_float)
    ren2.background((0, 0, 0.))

    report = window.analyze_scene(ren2)
    npt.assert_equal(report.bg_color, (0, 0, 0))

    ren2.add(axes)

    report = window.analyze_scene(ren2)
    npt.assert_equal(report.actors, 3)

    window.rm(ren2, axes)
    report = window.analyze_scene(ren2)
    npt.assert_equal(report.actors, 0)


@npt.dec.skipif(skip_it)
@xvfb_it
def test_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        scene = window.Renderer()
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        scene = window.renderer()
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        scene = window.ren()
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)


@npt.dec.skipif(skip_it)
@xvfb_it
def test_active_camera():
    scene = window.Scene()
    scene.add(actor.axes(scale=(1, 1, 1)))

    scene.reset_camera()
    scene.reset_clipping_range()

    direction = scene.camera_direction()
    position, focal_point, view_up = scene.get_camera()

    scene.set_camera((0., 0., 1.), (0., 0., 0), view_up)

    position, focal_point, view_up = scene.get_camera()
    npt.assert_almost_equal(np.dot(direction, position), -1)

    scene.zoom(1.5)

    new_position, _, _ = scene.get_camera()

    npt.assert_array_almost_equal(position, new_position)

    scene.zoom(1)

    # rotate around focal point
    scene.azimuth(90)

    position, _, _ = scene.get_camera()

    npt.assert_almost_equal(position, (1.0, 0.0, 0))

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.colors_found, [True])

    # rotate around camera's center
    scene.yaw(90)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=[(0, 0, 0)])
    npt.assert_equal(report.colors_found, [True])

    scene.yaw(-90)
    scene.elevation(90)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=(0, 255, 0))
    npt.assert_equal(report.colors_found, [True])

    scene.set_camera((0., 0., 1.), (0., 0., 0), view_up)

    # vertical rotation of the camera around the focal point
    scene.pitch(10)
    scene.pitch(-10)

    # rotate around the direction of projection
    scene.roll(90)

    # inverted normalized distance from focal point along the direction
    # of the camera

    position, _, _ = scene.get_camera()
    scene.dolly(0.5)
    new_position, _, _ = scene.get_camera()
    npt.assert_almost_equal(position[2], 0.5 * new_position[2])


@npt.dec.skipif(skip_it)
@xvfb_it
def test_parallel_projection():

    scene = window.Scene()
    axes = actor.axes()
    axes2 = actor.axes()
    axes2.SetPosition((2, 0, 0))

    # Add both axes.
    scene.add(axes, axes2)

    # Put the camera on a angle so that the
    # camera can show the difference between perspective
    # and parallel projection
    scene.set_camera((1.5, 1.5, 1.5))
    scene.GetActiveCamera().Zoom(2)

    # window.show(scene, reset_camera=True)
    scene.reset_camera()
    arr = window.snapshot(scene)

    scene.projection('parallel')
    # window.show(scene, reset_camera=False)
    arr2 = window.snapshot(scene)
    # Because of the parallel projection the two axes
    # will have the same size and therefore occupy more
    # pixels rather than in perspective projection were
    # the axes being further will be smaller.
    npt.assert_equal(np.sum(arr2 > 0) > np.sum(arr > 0), True)


@npt.dec.skipif(skip_it)
@xvfb_it
def test_order_transparent():

    scene = window.Scene()

    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0., .5, 0.]])
    stream_actor = actor.streamtube(lines, colors, linewidth=0.3, opacity=0.5)

    # scene.add(stream_actor)

    # scene.reset_camera()

    # # green in front
    # scene.elevation(90)
    # scene.camera().OrthogonalizeViewUp()
    # scene.reset_clipping_range()

    # scene.reset_camera()

    # not_xvfb = os.environ.get("TEST_WITH_XVFB", False)

    # if not_xvfb:
    #     arr = window.snapshot(scene, fname='green_front.png',
    #                           offscreen=True, order_transparent=False)
    # else:
    #     arr = window.snapshot(scene, fname='green_front.png',
    #                           offscreen=False, order_transparent=False)

    # # therefore the green component must have a higher value (in RGB terms)
    # npt.assert_equal(arr[150, 150][1] > arr[150, 150][0], True)

    # # red in front
    # scene.elevation(-180)
    # scene.camera().OrthogonalizeViewUp()
    # scene.reset_clipping_range()

    # if not_xvfb:
    #     arr = window.snapshot(scene, fname='red_front.png',
    #                           offscreen=True, order_transparent=True)
    # else:
    #     arr = window.snapshot(scene, fname='red_front.png',
    #                           offscreen=False, order_transparent=True)

    # # therefore the red component must have a higher value (in RGB terms)
    # npt.assert_equal(arr[150, 150][0] > arr[150, 150][1], True)


if __name__ == '__main__':
    # npt.run_module_suite()
    test_deprecated()
