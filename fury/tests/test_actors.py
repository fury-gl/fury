import os
import itertools
from tempfile import mkstemp, TemporaryDirectory as InTemporaryDirectory

import pytest
import numpy as np
import numpy.testing as npt
from scipy.ndimage.measurements import center_of_mass

from fury import shaders
from fury import actor, window
from fury.actor import grid
from fury.decorators import skip_osx, skip_win
from fury.utils import shallow_copy, rotate
from fury.testing import assert_greater, assert_greater_equal

# Allow import, but disable doctests if we don't have dipy
from fury.optpkg import optional_package
dipy, have_dipy, _ = optional_package('dipy')
matplotlib, have_matplotlib, _ = optional_package('matplotlib')

if have_dipy:
    from dipy.data import get_sphere
    from dipy.tracking.streamline import (center_streamlines,
                                          transform_streamlines)
    from dipy.align.tests.test_streamlinear import fornix_streamlines
    from dipy.reconst.dti import color_fa, fractional_anisotropy

if have_matplotlib:
    import matplotlib.pyplot as plt
    from fury.convert import matplotlib_figure_to_numpy


def test_slicer(verbose=False):
    scene = window.Scene()
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine, value_range=[data.min(), data.max()])
    slicer.display(None, None, 25)
    scene.add(slicer)

    scene.reset_camera()
    scene.reset_clipping_range()
    # window.show(scene)

    # copy pixels in numpy array directly
    arr = window.snapshot(scene, 'test_slicer.png', offscreen=True)

    if verbose:
        print(arr.sum())
        print(np.sum(arr == 0))
        print(np.sum(arr > 0))
        print(arr.shape)
        print(arr.dtype)

    report = window.analyze_snapshot(arr, find_objects=True)

    npt.assert_equal(report.objects, 1)
    # print(arr[..., 0])

    # The slicer can cut directly a smaller part of the image
    slicer.display_extent(10, 30, 10, 30, 35, 35)
    scene.ResetCamera()

    scene.add(slicer)

    # save pixels in png file not a numpy array
    with InTemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'slice.png')
        window.snapshot(scene, fname, offscreen=True)
        report = window.analyze_snapshot(fname, find_objects=True)
        npt.assert_equal(report.objects, 1)

    # Test Errors
    data_4d = (255 * np.random.rand(50, 50, 50, 50))
    npt.assert_raises(ValueError, actor.slicer, data_4d)
    npt.assert_raises(ValueError, actor.slicer, np.ones(10))

    scene.clear()

    rgb = np.zeros((30, 30, 30, 3), dtype='f8')
    rgb[..., 0] = 255
    rgb_actor = actor.slicer(rgb)

    scene.add(rgb_actor)

    scene.reset_camera()
    scene.reset_clipping_range()

    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True])

    lut = actor.colormap_lookup_table(scale_range=(0, 255),
                                      hue_range=(0.4, 1.),
                                      saturation_range=(1, 1.),
                                      value_range=(0., 1.))
    scene.clear()
    slicer_lut = actor.slicer(data, lookup_colormap=lut)

    slicer_lut.display(10, None, None)
    slicer_lut.display(None, 10, None)
    slicer_lut.display(None, None, 10)

    slicer_lut.opacity(0.5)
    slicer_lut.tolerance(0.03)
    slicer_lut2 = slicer_lut.copy()
    npt.assert_equal(slicer_lut2.GetOpacity(), 0.5)
    npt.assert_equal(slicer_lut2.picker.GetTolerance(), 0.03)
    slicer_lut2.opacity(1)
    slicer_lut2.tolerance(0.025)
    slicer_lut2.display(None, None, 10)
    scene.add(slicer_lut2)

    scene.reset_clipping_range()

    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)

    scene.clear()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])
    slicer = actor.slicer(data, affine, interpolation='nearest')
    slicer.display(None, None, 25)

    scene.add(slicer)
    scene.reset_camera()
    scene.reset_clipping_range()

    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(data.shape, slicer.shape)
    slicer2 = slicer.copy()
    npt.assert_equal(slicer2.shape, slicer.shape)

    scene.clear()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])

    from dipy.align.reslice import reslice

    data2, affine2 = reslice(data, affine, zooms=(1, 3, 2),
                             new_zooms=(1, 1, 1))

    slicer = actor.slicer(data2, affine2, interpolation='linear')
    slicer.display(None, None, 25)

    scene.add(slicer)
    scene.reset_camera()
    scene.reset_clipping_range()

    # window.show(scene, reset_camera=False)
    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_array_equal([1, 3, 2] * np.array(data.shape),
                           np.array(slicer.shape))


def test_surface():
    import math
    import random
    from scipy.spatial import Delaunay
    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = - math.sin(i) * math.cos(j)
            fact2 = - math.exp(abs(1 - math.sqrt(i ** 2 + j ** 2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])

    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices

    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, "butterfly", "loop"]

    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                scene = window.Scene(background=(1, 1, 1))
                surface_actor = actor.surface(vertices, faces=face,
                                              colors=color, smooth=smooth_type)
                scene.add(surface_actor)
                # window.show(scene, size=(600, 600), reset_camera=False)
                arr = window.snapshot(scene, 'test_surface.png',
                                      offscreen=True)
                report = window.analyze_snapshot(arr, find_objects=True)
                npt.assert_equal(report.objects, 1)


def test_contour_from_roi(interactive=False):

    # Render volume
    scene = window.Scene()
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.
    data[25, 20:30, 25] = 1.
    affine = np.eye(4)
    surface = actor.contour_from_roi(data, affine,
                                     color=np.array([1, 0, 1]),
                                     opacity=.5)
    scene.add(surface)

    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test Errors
    npt.assert_raises(ValueError, actor.contour_from_roi, np.ones(50))

    # Test binarization
    scene2 = window.Scene()
    data2 = np.zeros((50, 50, 50))
    data2[20:30, 25, 25] = 1.
    data2[35:40, 25, 25] = 1.
    affine = np.eye(4)
    surface2 = actor.contour_from_roi(data2, affine,
                                      color=np.array([0, 1, 1]),
                                      opacity=.5)
    scene2.add(surface2)

    scene2.reset_camera()
    scene2.reset_clipping_range()
    if interactive:
        window.show(scene2)

    arr = window.snapshot(scene, 'test_surface.png', offscreen=True)
    arr2 = window.snapshot(scene2, 'test_surface2.png', offscreen=True)

    report = window.analyze_snapshot(arr, find_objects=True)
    report2 = window.analyze_snapshot(arr2, find_objects=True)

    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report2.objects, 2)

    # test on real streamlines using tracking example
    if have_dipy:
        from dipy.data import read_stanford_labels
        from dipy.reconst.shm import CsaOdfModel
        from dipy.data import default_sphere
        from dipy.direction import peaks_from_model
        from fury.colormap import line_colors
        from dipy.tracking import utils
        try:
            from dipy.tracking.local import ThresholdTissueClassifier \
                as ThresholdStoppingCriterion
            from dipy.tracking.local import LocalTracking
        except ImportError:
            from dipy.tracking.stopping_criterion import \
                ThresholdStoppingCriterion
            from dipy.tracking.local_tracking import LocalTracking

        hardi_img, gtab, labels_img = read_stanford_labels()
        data = np.asanyarray(hardi_img.dataobj)
        labels = np.asanyarray(labels_img.dataobj)
        affine = hardi_img.affine

        white_matter = (labels == 1) | (labels == 2)

        csa_model = CsaOdfModel(gtab, sh_order=6)
        csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                                     relative_peak_threshold=.8,
                                     min_separation_angle=45,
                                     mask=white_matter)

        classifier = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

        seed_mask = labels == 2
        seeds = utils.seeds_from_mask(seed_mask, density=[1, 1, 1],
                                      affine=affine)

        # Initialization of LocalTracking.
        # The computation happens in the next step.
        streamlines = LocalTracking(csa_peaks, classifier, seeds, affine,
                                    step_size=2)

        # Compute streamlines and store as a list.
        streamlines = list(streamlines)

        # Prepare the display objects.
        streamlines_actor = actor.line(streamlines, line_colors(streamlines))
        seedroi_actor = actor.contour_from_roi(seed_mask, affine,
                                               [0, 1, 1], 0.5)

        # Create the 3d display.
        r = window.Scene()
        r2 = window.Scene()
        r.add(streamlines_actor)
        arr3 = window.snapshot(r, 'test_surface3.png', offscreen=True)
        report3 = window.analyze_snapshot(arr3, find_objects=True)
        r2.add(streamlines_actor)
        r2.add(seedroi_actor)
        arr4 = window.snapshot(r2, 'test_surface4.png', offscreen=True)
        report4 = window.analyze_snapshot(arr4, find_objects=True)

        # assert that the seed ROI rendering is not far
        # away from the streamlines (affine error)
        npt.assert_equal(report3.objects, report4.objects)
        # window.show(r)
        # window.show(r2)


@pytest.mark.skipif(skip_osx, reason="This test does not work on macOS + "
                                     "Travis. It works on a local machine"
                                     " with 3 different version of VTK. There"
                                     " are 2 problems to check: Travis macOS"
                                     " vs Azure macOS and an issue with"
                                     " vtkAssembly + actor opacity.")
def test_contour_from_label(interactive=False):

    # Render volumne
    scene = window.Scene()
    data = np.zeros((50, 50, 50))
    data[5:15, 1:10, 25] = 1.
    data[25:35, 1:10, 25] = 2.
    data[40:49, 1:10, 25] = 3.

    color = np.array([[255, 0, 0, 0.6],
                      [0, 255, 0, 0.5],
                      [0, 0, 255, 1.0]])

    surface = actor.contour_from_label(data, color=color)

    scene.add(surface)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test Errors
    with npt.assert_raises(ValueError):
        actor.contour_from_label(data, color=np.array([1, 2, 3]))
        actor.contour_from_label(np.ones(50))

    # Test binarization
    scene2 = window.Scene()
    data2 = np.zeros((50, 50, 50))
    data2[20:30, 25, 25] = 1.
    data2[25, 20:30, 25] = 2.

    color2 = np.array([[255, 0, 255],
                       [255, 255, 0]])

    surface2 = actor.contour_from_label(data2, color=color2)

    scene2.add(surface2)
    scene2.reset_camera()
    scene2.reset_clipping_range()
    if interactive:
        window.show(scene2)

    arr = window.snapshot(scene, 'test_surface.png', offscreen=True,
                          order_transparent=False)
    arr2 = window.snapshot(scene2, 'test_surface2.png', offscreen=True,
                           order_transparent=True)

    report = window.analyze_snapshot(arr, colors=[(255, 0, 0),
                                                  (0, 255, 0),
                                                  (0, 0, 255)],
                                     find_objects=True)
    report2 = window.analyze_snapshot(arr2, find_objects=True)

    npt.assert_equal(report.objects, 3)
    npt.assert_equal(report2.objects, 1)

    actor.contour_from_label(data)


def test_streamtube_and_line_actors():
    scene = window.Scene()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    line2 = line1 + np.array([0.5, 0., 0.])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    c = actor.line(lines, colors, linewidth=3)
    scene.add(c)

    c = actor.line(lines, colors, spline_subdiv=5, linewidth=3)
    scene.add(c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    scene.add(c2)

    arr = window.snapshot(scene)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    # as before with splines
    c2 = actor.streamtube(lines, colors, spline_subdiv=5, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    scene.add(c2)

    arr = window.snapshot(scene)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    c3 = actor.line(lines, colors, depth_cue=True, fake_tube=True)

    VTK_9_PLUS = window.vtk.vtkVersion.GetVTKMajorVersion() >= 9
    shader_obj = c3.GetShaderProperty() if VTK_9_PLUS else c3.GetMapper()
    mapper_code = shader_obj.GetGeometryShaderCode()
    file_code = shaders.load("line.geom")
    npt.assert_equal(mapper_code, file_code)

    npt.assert_equal(c3.GetProperty().GetRenderLinesAsTubes(), True)


@pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_bundle_maps():
    scene = window.Scene()
    bundle = fornix_streamlines()
    bundle, _ = center_streamlines(bundle)

    mat = np.array([[1, 0, 0, 100],
                    [0, 1, 0, 100],
                    [0, 0, 1, 100],
                    [0, 0, 0, 1.]])

    bundle = transform_streamlines(bundle, mat)

    # metric = np.random.rand(*(200, 200, 200))
    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(scale_range=(0., 100.),
                                      hue_range=(0., 0.1),
                                      saturation_range=(1, 1),
                                      value_range=(1., 1))

    line = actor.line(bundle, metric, linewidth=0.1, lookup_colormap=lut)
    scene.add(line)
    scene.add(actor.scalar_bar(lut, ' '))

    report = window.analyze_scene(scene)

    npt.assert_almost_equal(report.actors, 1)
    # window.show(scene)

    scene.clear()

    nb_points = np.sum([len(b) for b in bundle])
    values = 100 * np.random.rand(nb_points)
    # values[:nb_points/2] = 0

    line = actor.streamtube(bundle, values, linewidth=0.1, lookup_colormap=lut)
    scene.add(line)
    # window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    scene.clear()

    colors = np.random.rand(nb_points, 3)
    # values[:nb_points/2] = 0

    line = actor.line(bundle, colors, linewidth=2)
    scene.add(line)
    # window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
    # window.show(scene)

    arr = window.snapshot(scene)
    report2 = window.analyze_snapshot(arr)
    npt.assert_equal(report2.objects, 1)

    # try other input options for colors
    scene.clear()
    actor.line(bundle, (1., 0.5, 0))
    actor.line(bundle, np.arange(len(bundle)))
    actor.line(bundle)
    colors = [np.random.rand(*b.shape) for b in bundle]
    actor.line(bundle, colors=colors)


@pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_odf_slicer(interactive=False):

    sphere = get_sphere('symmetric362')

    shape = (11, 11, 11, sphere.vertices.shape[0])

    fid, fname = mkstemp(suffix='_odf_slicer.mmap')
    print(fid)
    print(fname)

    odfs = np.memmap(fname, dtype=np.float64, mode='w+',
                     shape=shape)

    odfs[:] = 1

    affine = np.eye(4)
    scene = window.Scene()

    mask = np.ones(odfs.shape[:3])
    mask[:4, :4, :4] = 0

    odfs[..., 0] = 1

    odf_actor = actor.odf_slicer(odfs, affine,
                                 mask=mask, sphere=sphere, scale=.25,
                                 colormap='blues')
    fa = 0. * np.zeros(odfs.shape[:3])
    fa[:, 0, :] = 1.
    fa[:, -1, :] = 1.
    fa[0, :, :] = 1.
    fa[-1, :, :] = 1.
    fa[5, 5, 5] = 1

    k = 5
    I, J, _ = odfs.shape[:3]

    fa_actor = actor.slicer(fa, affine)
    fa_actor.display_extent(0, I, 0, J, k, k)
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    odf_actor.display_extent(0, I, 0, J, k, k)
    odf_actor.GetProperty().SetOpacity(1.0)
    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 11 * 11)

    scene.clear()
    scene.add(fa_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    mask[:] = 0
    mask[5, 5, 5] = 1
    fa[5, 5, 5] = 0
    fa_actor = actor.slicer(fa, None)
    fa_actor.display(None, None, 5)
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='blues',
                                 norm=False, global_cm=True)
    scene.clear()
    scene.add(fa_actor)
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    scene.clear()
    scene.add(odf_actor)
    scene.add(fa_actor)
    odfs[:, :, :] = 1
    mask = np.ones(odfs.shape[:3])
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='blues',
                                 norm=False, global_cm=True)

    scene.clear()
    scene.add(odf_actor)
    scene.add(fa_actor)
    scene.add(actor.axes((11, 11, 11)))
    for i in range(11):
        odf_actor.display(i, None, None)
        fa_actor.display(i, None, None)
        if interactive:
            window.show(scene)
    for j in range(11):
        odf_actor.display(None, j, None)
        fa_actor.display(None, j, None)
        if interactive:
            window.show(scene)
    # with mask equal to zero everything should be black
    mask = np.zeros(odfs.shape[:3])
    odf_actor = actor.odf_slicer(odfs, None, mask=mask,
                                 sphere=sphere, scale=.25,
                                 colormap='blues',
                                 norm=False, global_cm=True)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    npt.assert_raises(IOError, actor.odf_slicer, odfs, None, mask=None,
                      sphere=sphere, scale=.25, colormap=None, norm=False,
                      global_cm=True)

    # colormap=None and global_cm=False results in directionally encoded colors
    scene.clear()
    scene.add(odf_actor)
    scene.add(fa_actor)
    odfs[:, :, :] = 1
    odf_actor = actor.odf_slicer(odfs, None, mask=None,
                                 sphere=sphere, scale=.25,
                                 colormap=None,
                                 norm=False, global_cm=False)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors, 1)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    del odf_actor
    odfs._mmap.close()
    del odfs
    os.close(fid)

    os.remove(fname)


def test_peak_slicer(interactive=False):

    _peak_dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f4')
    # peak_dirs.shape = (1, 1, 1) + peak_dirs.shape

    peak_dirs = np.zeros((11, 11, 11, 3, 3))

    peak_values = np.random.rand(11, 11, 11, 3)

    peak_dirs[:, :, :] = _peak_dirs

    scene = window.Scene()
    peak_actor = actor.peak_slicer(peak_dirs)
    scene.add(peak_actor)
    scene.add(actor.axes((11, 11, 11)))
    if interactive:
        window.show(scene)

    scene.clear()
    scene.add(peak_actor)
    scene.add(actor.axes((11, 11, 11)))
    for k in range(11):
        peak_actor.display_extent(0, 10, 0, 10, k, k)

    for j in range(11):
        peak_actor.display_extent(0, 10, j, j, 0, 10)

    for i in range(11):
        peak_actor.display(i, None, None)

    scene.rm_all()

    peak_actor = actor.peak_slicer(
        peak_dirs,
        peak_values,
        mask=None,
        affine=np.diag([3, 2, 1, 1]),
        colors=None,
        opacity=0.8,
        linewidth=3,
        lod=True,
        lod_points=10 ** 4,
        lod_points_size=3)

    scene.add(peak_actor)
    scene.add(actor.axes((11, 11, 11)))
    if interactive:
        window.show(scene)

    report = window.analyze_scene(scene)
    ex = ['vtkLODActor', 'vtkOpenGLActor']
    npt.assert_equal(report.actors_classnames, ex)

    # 6d data
    data_6d = (255 * np.random.rand(5, 5, 5, 5, 5, 5))
    npt.assert_raises(ValueError, actor.peak_slicer, data_6d, data_6d)


@pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_tensor_slicer(interactive=False):

    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((3, 2, 4, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    sphere = get_sphere('symmetric724')

    affine = np.eye(4)
    scene = window.Scene()

    tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                       sphere=sphere, scale=.3, opacity=0.4)
    _, J, K = mevals.shape[:3]
    scene.add(tensor_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    tensor_actor.display_extent(0, 1, 0, J, 0, K)
    if interactive:
        window.show(scene, reset_camera=False)

    tensor_actor.GetProperty().SetOpacity(1.0)
    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

    # Test extent
    big_extent = scene.GetActors().GetLastActor().GetBounds()
    big_extent_x = abs(big_extent[1] - big_extent[0])
    tensor_actor.display(x=2)

    if interactive:
        window.show(scene, reset_camera=False)

    small_extent = scene.GetActors().GetLastActor().GetBounds()
    small_extent_x = abs(small_extent[1] - small_extent[0])
    npt.assert_equal(big_extent_x > small_extent_x, True)

    # Test empty mask
    empty_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                      mask=np.zeros(mevals.shape[:3]),
                                      sphere=sphere, scale=.3)
    npt.assert_equal(empty_actor.GetMapper(), None)

    # Test mask
    mask = np.ones(mevals.shape[:3])
    mask[:2, :3, :3] = 0
    cfa = color_fa(fractional_anisotropy(mevals), mevecs)
    tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                       mask=mask, scalar_colors=cfa,
                                       sphere=sphere, scale=.3)
    scene.clear()
    scene.add(tensor_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    mask_extent = scene.GetActors().GetLastActor().GetBounds()
    mask_extent_x = abs(mask_extent[1] - mask_extent[0])
    npt.assert_equal(big_extent_x > mask_extent_x, True)

    # test display
    tensor_actor.display()
    current_extent = scene.GetActors().GetLastActor().GetBounds()
    current_extent_x = abs(current_extent[1] - current_extent[0])
    npt.assert_equal(big_extent_x > current_extent_x, True)
    if interactive:
        window.show(scene, reset_camera=False)

    tensor_actor.display(y=1)
    current_extent = scene.GetActors().GetLastActor().GetBounds()
    current_extent_y = abs(current_extent[3] - current_extent[2])
    big_extent_y = abs(big_extent[3] - big_extent[2])
    npt.assert_equal(big_extent_y > current_extent_y, True)
    if interactive:
        window.show(scene, reset_camera=False)

    tensor_actor.display(z=1)
    current_extent = scene.GetActors().GetLastActor().GetBounds()
    current_extent_z = abs(current_extent[5] - current_extent[4])
    big_extent_z = abs(big_extent[5] - big_extent[4])
    npt.assert_equal(big_extent_z > current_extent_z, True)
    if interactive:
        window.show(scene, reset_camera=False)

    # Test error handling of the method when
    # incompatible dimension of mevals and evecs are passed.
    mevals = np.zeros((3, 2, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    with npt.assert_raises(RuntimeError):
        tensor_actor = actor.tensor_slicer(mevals, mevecs, affine=affine,
                                           mask=mask, scalar_colors=cfa,
                                           sphere=sphere, scale=.3)


def test_dots(interactive=False):
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])

    dots_actor = actor.dots(points, color=(0, 255, 0))

    scene = window.Scene()
    scene.add(dots_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

    extent = scene.GetActors().GetLastActor().GetBounds()
    npt.assert_equal(extent, (0.0, 1.0, 0.0, 1.0, 0.0, 0.0))

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr,
                                     colors=(0, 255, 0))
    npt.assert_equal(report.objects, 3)

    # Test one point
    points = np.array([0, 0, 0])
    dot_actor = actor.dots(points, color=(0, 0, 255))

    scene.clear()
    scene.add(dot_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr,
                                     colors=(0, 0, 255))
    npt.assert_equal(report.objects, 1)


def test_points(interactive=False):
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    points_actor = actor.point(points, colors)

    scene = window.Scene()
    scene.add(points_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr,
                                     colors=colors)
    npt.assert_equal(report.objects, 3)


def test_labels(interactive=False):

    text_actor = actor.label("Hello")

    scene = window.Scene()
    scene.add(text_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)


def test_spheres(interactive=False):

    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])

    scene = window.Scene()
    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3])
    scene.add(sphere_actor)

    if interactive:
        window.show(scene, order_transparent=True)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr,
                                     colors=colors)
    npt.assert_equal(report.objects, 3)

    # test with an unique color for all centers
    scene.clear()
    sphere_actor = actor.sphere(centers=xyzr[:, :3],
                                colors=np.array([1, 0, 0]),
                                radii=xyzr[:, 3])
    scene.add(sphere_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=(1, 0, 0))
    npt.assert_equal(report.colors_found, [True])


def test_cones_vertices_faces(interactive=False):

    scene = window.Scene()
    centers = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    faces = np.array([[0, 1, 3], [0, 1, 2]])
    cone_actor = actor.cone(centers=centers, directions=directions,
                            colors=colors[:], vertices=vertices,
                            faces=faces)
    scene.add(cone_actor)

    if interactive:
        window.show(scene, order_transparent=True)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=[colors])
    npt.assert_equal(report.objects, 3)
    scene.clear()


def test_basic_geometry_actor(interactive=False):
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [1, 2, (1, 1, 1), [3, 2, 1], np.array([1, 2, 3]),
                  np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]])]

    actor_list = [[actor.cube, {}],
                  [actor.box, {}],
                  [actor.square, {}],
                  [actor.rectangle, {}],
                  [actor.frustum, {}],
                  [actor.octagonalprism, {}]]

    for act_func, extra_args in actor_list:
        for scale in scale_list:
            scene = window.Scene()
            g_actor = act_func(centers=centers, colors=colors,
                               directions=directions, scales=scale,
                               **extra_args)

            scene.add(g_actor)
            if interactive:
                window.show(scene)

            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr, colors=colors)
            msg = 'Failed with {}, scale={}'.format(act_func.__name__, scale)
            npt.assert_equal(report.objects, 3, err_msg=msg)


def test_advanced_geometry_actor(interactive=False):
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 0.5]])

    actor_list = [[actor.cone, {'directions': dirs, 'resolution': 8}],
                  [actor.arrow, {'directions': dirs, 'resolution': 9}],
                  [actor.cylinder, {'directions': dirs}]]

    scene = window.Scene()

    for act_func, extra_args in actor_list:
        colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])
        heights = np.array([5, 7, 10])

        geom_actor = act_func(centers=xyz, heights=heights, colors=colors[:],
                              **extra_args)
        scene.add(geom_actor)

        if interactive:
            window.show(scene, order_transparent=True)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=colors)
        npt.assert_equal(report.objects, 3)

        colors = np.array([1.0, 1.0, 1.0, 1.0])
        heights = 10

        scene.clear()
        geom_actor = act_func(centers=xyz[:, :3], heights=10, colors=colors[:],
                              **extra_args)
        scene.add(geom_actor)

        if interactive:
            window.show(scene, order_transparent=True)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=[colors])
        npt.assert_equal(report.objects, 3)

        scene.clear()


def test_text_3d():
    msg = 'I \nlove\n FURY'

    txt_actor = actor.text_3d(msg)
    npt.assert_equal(txt_actor.get_message().lower(), msg.lower())
    npt.assert_raises(ValueError, txt_actor.justification, 'middle')
    npt.assert_raises(ValueError, txt_actor.vertical_justification, 'center')

    scene = window.Scene()
    scene.add(txt_actor)
    txt_actor.vertical_justification('middle')
    txt_actor.justification('right')
    arr_right = window.snapshot(scene, size=(1920, 1080), offscreen=True)
    scene.clear()
    txt_actor.vertical_justification('middle')
    txt_actor.justification('left')
    scene.add(txt_actor)
    arr_left = window.snapshot(scene, size=(1920, 1080), offscreen=True)
    # X axis of right alignment should have a lower center of mass position
    # than left
    assert_greater(center_of_mass(arr_left)[0], center_of_mass(arr_right)[0])
    scene.clear()
    txt_actor.justification('center')
    txt_actor.vertical_justification('top')
    scene.add(txt_actor)
    arr_top = window.snapshot(scene, size=(1920, 1080), offscreen=True)
    scene.clear()
    txt_actor.justification('center')
    txt_actor.vertical_justification('bottom')
    scene.add(txt_actor)
    arr_bottom = window.snapshot(scene, size=(1920, 1080), offscreen=True)
    assert_greater_equal(center_of_mass(arr_bottom)[0],
                         center_of_mass(arr_top)[0])

    scene.clear()
    txt_actor.font_style(bold=True, italic=True, shadow=True)
    scene.add(txt_actor)
    arr = window.snapshot(scene, size=(1920, 1080), offscreen=True)
    assert_greater_equal(arr.mean(), arr_bottom.mean())


def test_container():
    container = actor.Container()

    axes = actor.axes()
    container.add(axes)
    npt.assert_equal(len(container), 1)
    npt.assert_equal(container.GetBounds(), axes.GetBounds())
    npt.assert_equal(container.GetCenter(), axes.GetCenter())
    npt.assert_equal(container.GetLength(), axes.GetLength())

    container.clear()
    npt.assert_equal(len(container), 0)

    container.add(axes)
    container_shallow_copy = shallow_copy(container)
    container_shallow_copy.add(actor.axes())

    assert_greater(len(container_shallow_copy), len(container))
    npt.assert_equal(container_shallow_copy.GetPosition(),
                     container.GetPosition())
    npt.assert_equal(container_shallow_copy.GetVisibility(),
                     container.GetVisibility())

    # Check is the shallow_copy do not modify original container
    container_shallow_copy.SetVisibility(False)
    npt.assert_equal(container.GetVisibility(), True)

    container_shallow_copy.SetPosition((1, 1, 1))
    npt.assert_equal(container.GetPosition(), (0, 0, 0))


def test_grid(_interactive=False):

    vol1 = np.zeros((100, 100, 100))
    vol1[25:75, 25:75, 25:75] = 100
    contour_actor1 = actor.contour_from_roi(vol1, np.eye(4),
                                            (1., 0, 0), 1.)

    vol2 = np.zeros((100, 100, 100))
    vol2[25:75, 25:75, 25:75] = 100

    contour_actor2 = actor.contour_from_roi(vol2, np.eye(4),
                                            (1., 0.5, 0), 1.)
    vol3 = np.zeros((100, 100, 100))
    vol3[25:75, 25:75, 25:75] = 100

    contour_actor3 = actor.contour_from_roi(vol3, np.eye(4),
                                            (1., 0.5, 0.5), 1.)

    scene = window.Scene()
    actors = []
    texts = []

    actors.append(contour_actor1)
    text_actor1 = actor.text_3d('cube 1', justification='center')
    texts.append(text_actor1)

    actors.append(contour_actor2)
    text_actor2 = actor.text_3d('cube 2', justification='center')
    texts.append(text_actor2)

    actors.append(contour_actor3)
    text_actor3 = actor.text_3d('cube 3', justification='center')
    texts.append(text_actor3)

    actors.append(shallow_copy(contour_actor1))
    text_actor1 = 'cube 4'
    texts.append(text_actor1)

    actors.append(shallow_copy(contour_actor2))
    text_actor2 = 'cube 5'
    texts.append(text_actor2)

    actors.append(shallow_copy(contour_actor3))
    text_actor3 = 'cube 6'
    texts.append(text_actor3)

    # show the grid without the captions
    container = grid(actors=actors, captions=None,
                     caption_offset=(0, -40, 0),
                     cell_padding=(10, 10), dim=(2, 3))

    scene.add(container)

    scene.projection('orthogonal')

    counter = itertools.count()

    show_m = window.ShowManager(scene)

    show_m.initialize()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        # show_m.scene.zoom(1)
        show_m.render()
        if cnt == 4:
            show_m.exit()
            show_m.destroy_timers()

    show_m.add_timer_callback(True, 200, timer_callback)
    show_m.start()

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 6)

    scene.rm_all()

    counter = itertools.count()
    show_m = window.ShowManager(scene)
    show_m.initialize()
    # show the grid with the captions
    container = grid(actors=actors, captions=texts,
                     caption_offset=(0, -50, 0),
                     cell_padding=(10, 10),
                     dim=(3, 3))

    scene.add(container)

    show_m.add_timer_callback(True, 200, timer_callback)
    show_m.start()

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects > 6, True)


def test_direct_sphere_mapping():

    arr = 255 * np.ones((810, 1620, 3), dtype='uint8')
    rows, cols, _ = arr.shape

    rs = rows // 2
    cs = cols // 2
    w = 150 // 2

    arr[rs - w: rs + w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    # enable to see pacman on sphere
    # arr[0: 2 * w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    scene = window.Scene()
    tsa = actor.texture_on_sphere(arr)
    scene.add(tsa)
    rotate(tsa, rotation=(90, 0, 1, 0))
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(255, 127, 0)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True])


def test_texture_mapping():

    arr = np.zeros((512, 212, 3), dtype='uint8')
    arr[:256, :] = np.array([255, 0, 0])
    arr[256:, :] = np.array([0, 255, 0])
    tp = actor.texture(arr,
                       interp=True)
    scene = window.Scene()
    scene.add(tp)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(255, 0, 0), (0, 255, 0)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True])


def test_figure_vs_texture_actor():

    arr = (255 * np.ones((512, 212, 4))).astype('uint8')

    arr[20:40, 20:40, 3] = 0
    tp = actor.figure(arr)
    arr[20:40, 20:40, :] = np.array([255, 0, 0, 255], dtype='uint8')
    tp2 = actor.texture(arr)
    scene = window.Scene()
    scene.add(tp)
    scene.add(tp2)
    tp2.SetPosition(0, 0, -50)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(display, bg_color=(0, 0, 0),
                                  colors=[(255, 0, 0), (255, 255, 255)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True])


@pytest.mark.skipif(not have_matplotlib, reason="Requires MatplotLib")
def test_matplotlib_figure():

    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    fig = plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Categorical Plotting')

    arr = matplotlib_figure_to_numpy(fig, dpi=500, transparent=True)
    fig_actor = actor.figure(arr, 'cubic')
    fig_actor2 = actor.figure(arr, 'cubic')
    scene = window.Scene()
    scene.background((1, 1, 1.))

    ax_actor = actor.axes(scale=(1000, 1000, 1000))
    scene.add(ax_actor)
    scene.add(fig_actor)
    scene.add(fig_actor2)
    ax_actor.SetPosition(0, 500, -800)
    fig_actor2.SetPosition(500, 800, -400)
    display = window.snapshot(scene, 'test_mpl.png', order_transparent=False,
                              offscreen=True)
    res = window.analyze_snapshot(display, bg_color=(255, 255, 255.),
                                  colors=[(31, 119, 180), (255, 0, 0)],
                                  find_objects=False)
    npt.assert_equal(res.colors_found, [True, True])


def test_superquadric_actor(interactive=False):
    scene = window.Scene()
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.random.rand(3, 3)
    scales = [1, 2, 3]
    roundness = np.array([[1, 1], [1, 2], [2, 1]])

    sq_actor = actor.superquadric(centers, roundness=roundness,
                                  directions=directions,
                                  colors=colors.astype(np.uint8),
                                  scales=scales)
    scene.add(sq_actor)
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene, offscreen=True)
    arr[arr > 0] = 255  # Normalization
    report = window.analyze_snapshot(arr, colors=255*colors.astype(np.uint8))
    npt.assert_equal(report.objects, 3)
    npt.assert_equal(report.colors_found, [True, True, True])


def test_billboard_actor(interactive=False):
    scene = window.Scene()
    scene.background((1, 1, 1))
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    scales = [1, 2, 1]

    fake_sphere = \
        """
        float len = length(point);
        float radius = 1.;
        if(len > radius)
            {discard;}

        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df_1 = max(0, dot(direction, normalizedPoint));
        float sf_1 = pow(df_1, 24);
        fragOutput0 = vec4(max(df_1 * color, sf_1 * vec3(1)), 1);
        """

    billboard_actor = actor.billboard(centers,
                                      colors=colors.astype(np.uint8),
                                      scales=scales,
                                      fs_impl=fake_sphere)
    scene.add(billboard_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)


@pytest.mark.skipif(skip_win, reason="This test does not work on Windows"
                                     " due to snapshot (memory access"
                                     " violation). Check what is causing this"
                                     " issue with shader")
def test_sdf_actor(interactive=False):
    scene = window.Scene()
    scene.background((1, 1, 1))
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    scales = [1, 2, 3]
    primitive = ['sphere', 'ellipsoid', 'torus']

    sdf_actor = actor.sdf(centers, directions,
                          colors, primitive, scales)
    scene.add(sdf_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 3)
