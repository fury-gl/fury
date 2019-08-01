import os
import platform
import warnings
import numpy as np
from fury import actor, window, io
import numpy.testing as npt
from fury.testing import captured_output, assert_less_equal
from fury.decorators import xvfb_it
from fury.tmpdirs import InTemporaryDirectory

skip_osx = platform.system().lower() == "darwin"
skip_win = platform.system().lower() == "windows"


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

    scene.add(axes)
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

    ren2.rm(axes)
    report = window.analyze_scene(ren2)
    npt.assert_equal(report.actors, 0)

    with captured_output() as (out, err):
        scene.camera_info()
    npt.assert_equal(out.getvalue().strip(),
                     '# Active Camera\n   '
                     'Position (0.00, 0.00, 1.00)\n   '
                     'Focal Point (0.00, 0.00, 0.00)\n   '
                     'View Up (0.00, 1.00, 0.00)')
    npt.assert_equal(err.getvalue().strip(), '')


@xvfb_it
def test_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", PendingDeprecationWarning)
        scene = window.Renderer()
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, PendingDeprecationWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", PendingDeprecationWarning)
        scene = window.renderer(background=(0.0, 1.0, 0.0))
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, PendingDeprecationWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", PendingDeprecationWarning)
        scene = window.ren()
        npt.assert_equal(scene.size(), (0, 0))
        assert len(w) == 1
        assert issubclass(w[-1].category, PendingDeprecationWarning)

    scene = window.Scene()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", PendingDeprecationWarning)
        obj = actor.axes(scale=(1, 1, 1))
        window.add(scene, obj)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 3)
        window.rm(scene, obj)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 0)
        window.add(scene, obj)
        window.rm_all(scene)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr)
        npt.assert_equal(report.objects, 0)
        window.add(scene, obj)
        window.clear(scene)
        report = window.analyze_renderer(scene)
        npt.assert_equal(report.actors, 0)
        npt.assert_equal(len(w), 6)
        assert issubclass(w[-1].category, PendingDeprecationWarning)


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
    new_position, focal_point, view_up = scene.get_camera()
    npt.assert_almost_equal(position[2], 0.5 * new_position[2])

    cam = scene.camera()
    npt. assert_equal(new_position, cam.GetPosition())
    npt. assert_equal(focal_point, cam.GetFocalPoint())
    npt. assert_equal(view_up, cam.GetViewUp())


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
    scene.projection('perspective')
    arr2 = window.snapshot(scene)
    npt.assert_equal(np.sum(arr2 > 0), np.sum(arr > 0))


@npt.dec.skipif(skip_osx or skip_win)
@xvfb_it
def test_order_transparent():

    scene = window.Scene()

    lines = [np.array([[1, 0, 1.], [-1, 0, 1.]]),
             np.array([[1, 0, -1.], [-1, 0, -1.]])]
    colors = np.array([[1., 0., 0.], [0., 1., 0.]])
    stream_actor = actor.streamtube(lines, colors, linewidth=0.3, opacity=0.5)

    scene.add(stream_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    not_xvfb = os.environ.get("TEST_WITH_XVFB", False)

    if not_xvfb:
        arr = window.snapshot(scene, fname='green_front.png',
                              offscreen=True, order_transparent=False)
    else:
        arr = window.snapshot(scene, fname='green_front.png',
                              offscreen=False, order_transparent=False)

    green_no_ot = arr[150, 150, 1]

    if not_xvfb:
        arr = window.snapshot(scene, fname='red_front.png',
                              offscreen=True, order_transparent=True)
    else:
        arr = window.snapshot(scene, fname='red_front.png',
                              offscreen=False, order_transparent=True)

    # when order transparency is True green should be weaker
    green_ot = arr[150, 150, 1]

    npt.assert_equal(green_no_ot > green_ot, True)


@xvfb_it
def test_stereo():

    scene = window.Scene()

    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0., 1., 0.]])
    stream_actor = actor.streamtube(lines, colors, linewidth=0.3, opacity=0.5)

    scene.add(stream_actor)

    # green in front
    scene.elevation(90)
    scene.camera().OrthogonalizeViewUp()
    scene.reset_clipping_range()

    scene.reset_camera()

    not_xvfb = os.environ.get("TEST_WITH_XVFB", False)

    if not_xvfb:
        mono = window.snapshot(scene, fname='stereo_off.png', offscreen=True,
                               size=(300, 300), order_transparent=True,
                               stereo='off')
    else:
        mono = window.snapshot(scene, fname='stereo_off.png', offscreen=False,
                               size=(300, 300), order_transparent=True,
                               stereo='off')

    if not_xvfb:
        stereo = window.snapshot(scene, fname='stereo_horizontal.png',
                                 offscreen=True, size=(300, 300),
                                 order_transparent=True, stereo='On')
    else:
        stereo = window.snapshot(scene, fname='stereo_horizontal.png',
                                 offscreen=False, size=(300, 300),
                                 order_transparent=True, stereo='On')

    # mono render should have values in the center
    # horizontal split stereo render should be empty in the center
    npt.assert_raises(AssertionError, npt.assert_array_equal,
                      mono[150, 150], [0, 0, 0])
    npt.assert_array_equal(stereo[150, 150], [0, 0, 0])


@npt.dec.skipif(skip_osx)
@xvfb_it
def test_record():
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1., 1]])
    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3])
    scene = window.Scene()
    scene.add(sphere_actor)

    def test_content(filename='fury.png', colors_found=(True, True)):
        npt.assert_equal(os.path.isfile(filename), True)
        arr = io.load_image(filename)
        report = window.analyze_snapshot(arr, colors=[(0, 255, 0),
                                                      (255, 0, 0)])
        npt.assert_equal(report.objects, 3)
        npt.assert_equal(report.colors_found, colors_found)
        return arr

    # Basic test
    with InTemporaryDirectory():
        window.record(scene)
        test_content()

    # test out_path and path_numbering, n_frame
    with InTemporaryDirectory():
        filename = "tmp_snapshot.png"
        window.record(scene, out_path=filename)
        test_content(filename)
        window.record(scene, out_path=filename, path_numbering=True)
        test_content(filename + "000000.png")
        window.record(scene, out_path=filename, path_numbering=True,
                      n_frames=3)
        test_content(filename + "000000.png")
        test_content(filename + "000001.png")
        test_content(filename + "000002.png")
        npt.assert_equal(os.path.isfile(filename + "000003.png"), False)

    # test verbose
    with captured_output() as (out, _):
        window.record(scene, verbose=True)

    npt.assert_equal(out.getvalue().strip(),
                     "Camera Position (315.14, 0.00, 536.43)\n"
                     "Camera Focal Point (119.89, 0.00, 0.00)\n"
                     "Camera View Up (0.00, 1.00, 0.00)")
    # test camera option
    with InTemporaryDirectory():
        window.record(scene, cam_pos=(310, 0, 530), cam_focal=(120, 0, 0),
                      cam_view=(0, 0, 1))
        test_content()

    # test size and clipping
    with InTemporaryDirectory():
        window.record(scene, out_path='fury_1.png', size=(1000, 1000),
                      magnification=5)
        npt.assert_equal(os.path.isfile('fury_1.png'), True)
        arr = io.load_image('fury_1.png')

        npt.assert_equal(arr.shape, (5000, 5000, 3))

        window.record(scene, out_path='fury_2.png', size=(5000, 5000),
                      screen_clip=True)
        npt.assert_equal(os.path.isfile('fury_2.png'), True)
        arr = io.load_image('fury_2.png')

        assert_less_equal(arr.shape[0], 5000)
        assert_less_equal(arr.shape[1], 5000)


if __name__ == '__main__':
    npt.run_module_suite()
