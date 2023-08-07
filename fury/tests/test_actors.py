import itertools
import os
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import center_of_mass

from fury import actor
from fury import primitive as fp
from fury import shaders, window
from fury.actor import grid
from fury.decorators import skip_linux, skip_osx, skip_win
from fury.deprecator import ExpiredDeprecationError

# Allow import, but disable doctests if we don't have dipy
from fury.optpkg import optional_package
from fury.primitive import prim_sphere
from fury.testing import (
    assert_equal,
    assert_greater,
    assert_greater_equal,
    assert_less_equal,
    assert_not_equal,
)
from fury.utils import primitives_count_from_actor, rotate, shallow_copy

# dipy, have_dipy, _ = optional_package('dipy')
matplotlib, have_matplotlib, _ = optional_package('matplotlib')

# if have_dipy:
#     from dipy.data import get_sphere
#     from dipy.reconst.shm import sh_to_sf_matrix
#     from dipy.tracking.streamline import (center_streamlines,
#                                           transform_streamlines)
#     from dipy.align.tests.test_streamlinear import fornix_streamlines
#     from dipy.reconst.dti import color_fa, fractional_anisotropy

if have_matplotlib:
    import matplotlib.pyplot as plt

    from fury.convert import matplotlib_figure_to_numpy


class Sphere:

    vertices = None
    faces = None


def test_slicer(verbose=False):
    scene = window.Scene()
    data = 255 * np.random.rand(50, 50, 50)
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
    data_4d = 255 * np.random.rand(50, 50, 50, 50)
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

    lut = actor.colormap_lookup_table(
        scale_range=(0, 255),
        hue_range=(0.4, 1.0),
        saturation_range=(1, 1.0),
        value_range=(0.0, 1.0),
    )
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

    data = 255 * np.random.rand(50, 50, 50)
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


def test_surface():
    import math
    import random

    from scipy.spatial import Delaunay

    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = -math.sin(i) * math.cos(j)
            fact2 = -math.exp(abs(1 - math.sqrt(i**2 + j**2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])

    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices

    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, 'butterfly', 'loop']

    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                scene = window.Scene(background=(1, 1, 1))
                surface_actor = actor.surface(
                    vertices, faces=face, colors=color, smooth=smooth_type
                )
                scene.add(surface_actor)
                # window.show(scene, size=(600, 600), reset_camera=False)
                arr = window.snapshot(scene, 'test_surface.png', offscreen=True)
                report = window.analyze_snapshot(arr, find_objects=True)
                npt.assert_equal(report.objects, 1)


def test_contour_from_roi(interactive=False):

    # Render volume
    scene = window.Scene()
    data = np.zeros((50, 50, 50))
    data[20:30, 25, 25] = 1.0
    data[25, 20:30, 25] = 1.0
    affine = np.eye(4)
    surface = actor.contour_from_roi(
        data, affine, color=np.array([1, 0, 1]), opacity=0.5
    )
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
    data2[20:30, 25, 25] = 1.0
    data2[35:40, 25, 25] = 1.0
    affine = np.eye(4)
    surface2 = actor.contour_from_roi(
        data2, affine, color=np.array([0, 1, 1]), opacity=0.5
    )
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


@pytest.mark.skipif(
    skip_osx,
    reason='This test does not work on macOS + '
    'Travis. It works on a local machine'
    ' with 3 different version of VTK. There'
    ' are 2 problems to check: Travis macOS'
    ' vs Azure macOS and an issue with'
    ' vtkAssembly + actor opacity.',
)
def test_contour_from_label(interactive=False):
    # Render volume
    scene = window.Scene()
    data = np.zeros((50, 50, 50))
    data[5:15, 1:10, 25] = 1.0
    data[25:35, 1:10, 25] = 2.0
    data[40:49, 1:10, 25] = 3.0

    color = np.array([[255, 0, 0, 0.6], [0, 255, 0, 0.5], [0, 0, 255, 1.0]])

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
    data2[20:30, 25, 25] = 1.0
    data2[25, 20:30, 25] = 2.0

    color2 = np.array([[255, 0, 255], [255, 255, 0]])

    surface2 = actor.contour_from_label(data2, color=color2)

    scene2.add(surface2)
    scene2.reset_camera()
    scene2.reset_clipping_range()
    if interactive:
        window.show(scene2)

    arr = window.snapshot(
        scene, 'test_surface.png', offscreen=True, order_transparent=False
    )
    arr2 = window.snapshot(
        scene2, 'test_surface2.png', offscreen=True, order_transparent=True
    )

    report = window.analyze_snapshot(
        arr, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], find_objects=True
    )
    report2 = window.analyze_snapshot(arr2, find_objects=True)

    npt.assert_equal(report.objects, 3)
    npt.assert_equal(report2.objects, 1)

    actor.contour_from_label(data)


def test_streamtube_and_line_actors():
    scene = window.Scene()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.0]])
    line2 = line1 + np.array([0.5, 0.0, 0.0])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.0]])
    c = actor.line(lines, colors, linewidth=3)
    scene.add(c)

    c = actor.line(lines, colors, spline_subdiv=5, linewidth=3)
    scene.add(c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=0.1)
    c2.SetPosition(2, 0, 0)
    scene.add(c2)

    arr = window.snapshot(scene)

    report = window.analyze_snapshot(
        arr, colors=[(255, 0, 0), (0, 0, 255)], find_objects=True
    )

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    # as before with splines
    c2 = actor.streamtube(lines, colors, spline_subdiv=5, linewidth=0.1)
    c2.SetPosition(2, 0, 0)
    scene.add(c2)

    arr = window.snapshot(scene)

    report = window.analyze_snapshot(
        arr, colors=[(255, 0, 0), (0, 0, 255)], find_objects=True
    )

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    c3 = actor.line(lines, colors, depth_cue=True, fake_tube=True)

    shader_obj = c3.GetShaderProperty()
    mapper_code = shader_obj.GetGeometryShaderCode()
    file_code = shaders.import_fury_shader('line.geom')
    npt.assert_equal(mapper_code, file_code)

    npt.assert_equal(c3.GetProperty().GetRenderLinesAsTubes(), True)

    c4 = actor.streamtube(lines, colors, replace_strips=False)

    c5 = actor.streamtube(lines, colors, replace_strips=True)

    strips4 = c4.GetMapper().GetInput().GetStrips().GetData().GetSize()
    strips5 = c5.GetMapper().GetInput().GetStrips().GetData().GetSize()

    npt.assert_equal(strips4 > 0, True)
    npt.assert_equal(strips5 == 0, True)


def simulated_bundle(no_streamlines=10, waves=False):
    t = np.linspace(20, 80, 200)
    # parallel waves or parallel lines
    bundle = []
    for i in np.linspace(-5, 5, no_streamlines):
        if waves:
            pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        else:
            pts = np.vstack((np.zeros(t.shape), t, i * np.ones(t.shape))).T
        bundle.append(pts)

    return bundle


# @pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_bundle_maps():
    scene = window.Scene()
    bundle = simulated_bundle(no_streamlines=10, waves=False)

    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(
        scale_range=(0.0, 100.0),
        hue_range=(0.0, 0.1),
        saturation_range=(1, 1),
        value_range=(1.0, 1),
    )

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
    actor.line(bundle, (1.0, 0.5, 0))
    actor.line(bundle, np.arange(len(bundle)))
    actor.line(bundle)
    colors = [np.random.rand(*b.shape) for b in bundle]
    actor.line(bundle, colors=colors)


# @pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_odf_slicer(interactive=False):
    # TODO: we should change the odf_slicer to work directly
    # vertices and faces of a sphere rather that needing
    # a specific type of sphere. We can use prim_sphere
    # as an alternative to get_sphere.
    vertices, faces = prim_sphere('repulsion100', True)
    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    shape = (11, 11, 11, 100)
    odfs = np.ones(shape)

    affine = np.array(
        [
            [2.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mask = np.ones(odfs.shape[:3], bool)
    mask[:4, :4, :4] = False

    # Test that affine and mask work
    odf_actor = actor.odf_slicer(
        odfs, sphere=sphere, affine=affine, mask=mask, scale=0.25, colormap='blues'
    )

    k = 2
    I, J, _ = odfs.shape[:3]
    odf_actor.display_extent(0, I - 1, 0, J - 1, k, k)

    scene = window.Scene()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 11 * 11 - 16)

    # Test that global colormap works
    odf_actor = actor.odf_slicer(
        odfs,
        sphere=sphere,
        mask=mask,
        scale=0.25,
        colormap='blues',
        norm=False,
        global_cm=True,
    )
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that the most basic odf_slicer instantiation works
    odf_actor = actor.odf_slicer(odfs)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that odf_slicer.display works properly
    scene.clear()
    scene.add(odf_actor)
    scene.add(actor.axes((11, 11, 11)))
    for i in range(11):
        odf_actor.display(i, None, None)
        if interactive:
            window.show(scene)
    for j in range(11):
        odf_actor.display(None, j, None)
        if interactive:
            window.show(scene)

    # With mask equal to zero everything should be black
    mask = np.zeros(odfs.shape[:3])
    odf_actor = actor.odf_slicer(
        odfs,
        sphere=sphere,
        mask=mask,
        scale=0.25,
        colormap='blues',
        norm=False,
        global_cm=True,
    )
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # global_cm=True with colormap=None should raise an error
    npt.assert_raises(
        IOError,
        actor.odf_slicer,
        odfs,
        sphere=None,
        mask=None,
        scale=0.25,
        colormap=None,
        norm=False,
        global_cm=True,
    )

    vertices2, faces2 = prim_sphere('repulsion200', True)
    sphere2 = Sphere()
    sphere2.vertices = vertices2
    sphere2.faces = faces2

    # Dimension mismatch between sphere vertices and number
    # of SF coefficients will raise an error.
    npt.assert_raises(
        ValueError, actor.odf_slicer, odfs, mask=None, sphere=sphere2, scale=0.25
    )

    # colormap=None and global_cm=False results in directionally encoded colors
    odf_actor = actor.odf_slicer(
        odfs,
        sphere=None,
        mask=None,
        scale=0.25,
        colormap=None,
        norm=False,
        global_cm=False,
    )
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    del odf_actor
    del odfs


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

    peak_actor_sym = actor.peak_slicer(
        peak_dirs,
        peak_values,
        mask=None,
        affine=np.diag([3, 2, 1, 1]),
        colors=None,
        opacity=0.8,
        linewidth=3,
        lod=True,
        lod_points=10**4,
        lod_points_size=3,
    )

    peak_actor_asym = actor.peak_slicer(
        peak_dirs,
        peak_values,
        mask=None,
        affine=np.diag([3, 2, 1, 1]),
        colors=None,
        opacity=0.8,
        linewidth=3,
        lod=True,
        lod_points=10**4,
        lod_points_size=3,
        symmetric=False,
    )

    scene.add(peak_actor_sym)
    scene.add(peak_actor_asym)
    scene.add(actor.axes((11, 11, 11)))
    if interactive:
        window.show(scene)

    report = window.analyze_scene(scene)
    ex = ['vtkLODActor', 'vtkLODActor', 'vtkOpenGLActor']
    npt.assert_equal(report.actors_classnames, ex)

    # 6d data
    data_6d = 255 * np.random.rand(5, 5, 5, 5, 5, 5)
    npt.assert_raises(ValueError, actor.peak_slicer, data_6d, data_6d)


def test_peak():
    # 4D dirs data
    dirs_data_4d = np.random.rand(3, 4, 5, 6)
    npt.assert_raises(ValueError, actor.peak, dirs_data_4d)

    # 6D dirs data
    dirs_data_6d = np.random.rand(7, 8, 9, 10, 11, 12)
    npt.assert_raises(ValueError, actor.peak, dirs_data_6d)

    # 2D directions
    dirs_2d = np.random.rand(3, 4, 5, 6, 2)
    npt.assert_raises(ValueError, actor.peak, dirs_2d)

    # 4D directions
    dirs_4d = np.random.rand(3, 4, 5, 6, 4)
    npt.assert_raises(ValueError, actor.peak, dirs_4d)

    valid_dirs = np.random.rand(3, 4, 5, 6, 3)

    # 3D vals data
    vals_data_3d = np.random.rand(3, 4, 5)
    npt.assert_raises(ValueError, actor.peak, valid_dirs, peaks_values=vals_data_3d)

    # 5D vals data
    vals_data_5d = np.random.rand(6, 7, 8, 9, 10)
    npt.assert_raises(ValueError, actor.peak, valid_dirs, peaks_values=vals_data_5d)

    # Diff vals data #1
    vals_data_diff_1 = np.random.rand(3, 4, 5, 9)
    npt.assert_raises(ValueError, actor.peak, valid_dirs, peaks_values=vals_data_diff_1)

    # Diff vals data #2
    vals_data_diff_2 = np.random.rand(7, 8, 9, 10)
    npt.assert_raises(ValueError, actor.peak, valid_dirs, peaks_values=vals_data_diff_2)

    # 2D mask
    mask_2d = np.random.rand(2, 3)
    npt.assert_warns(UserWarning, actor.peak, valid_dirs, mask=mask_2d)

    # 4D mask
    mask_4d = np.random.rand(4, 5, 6, 7)
    npt.assert_warns(UserWarning, actor.peak, valid_dirs, mask=mask_4d)

    # Diff mask
    diff_mask = np.random.rand(6, 7, 8)
    npt.assert_warns(UserWarning, actor.peak, valid_dirs, mask=diff_mask)

    # Valid mask
    dirs000 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dirs100 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    peaks_dirs = np.empty((2, 1, 1, 3, 3))
    peaks_dirs[0, 0, 0, :, :] = dirs000
    peaks_dirs[1, 0, 0, :, :] = dirs100
    peaks_vals = np.ones((2, 1, 1, 3)) * 0.5
    mask = np.zeros((2, 1, 1))
    mask[0, 0, 0] = 1
    peaks_actor = actor.peak(peaks_dirs, peaks_values=peaks_vals, mask=mask)
    npt.assert_equal(peaks_actor.min_centers, [0, 0, 0])
    npt.assert_equal(peaks_actor.max_centers, [0, 0, 0])


# @pytest.mark.skipif(not have_dipy, reason="Requires DIPY")
def test_tensor_slicer(interactive=False):

    evals = np.array([1.4, 0.35, 0.35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((3, 2, 4, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    vertices, faces = prim_sphere('symmetric724', True)
    sphere = Sphere()
    sphere.vertices = vertices
    sphere.faces = faces

    affine = np.eye(4)
    scene = window.Scene()

    tensor_actor = actor.tensor_slicer(
        mevals, mevecs, affine=affine, sphere=sphere, scale=0.3, opacity=0.4
    )
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
    empty_actor = actor.tensor_slicer(
        mevals,
        mevecs,
        affine=affine,
        mask=np.zeros(mevals.shape[:3]),
        sphere=sphere,
        scale=0.3,
    )
    npt.assert_equal(empty_actor.GetMapper(), None)

    # Test error handling of the method when
    # incompatible dimension of mevals and evecs are passed.
    mevals = np.zeros((3, 2, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    with npt.assert_raises(RuntimeError):
        tensor_actor = actor.tensor_slicer(
            mevals,
            mevecs,
            affine=affine,
            mask=None,
            scalar_colors=None,
            sphere=sphere,
            scale=0.3,
        )
    # TODO: Add colorfa test here as previous test moved to DIPY.


def test_dot(interactive=False):
    # Test three points with different colors and opacities
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 0.5], [0, 0, 1, 0.3]])

    dots_actor = actor.dot(points, colors=colors)

    scene = window.Scene()
    scene.add(dots_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)

    arr = window.snapshot(scene)
    expected_colors = np.floor(colors[:, 3] * 255) * colors[:, :3]
    report = window.analyze_snapshot(arr, colors=expected_colors)
    npt.assert_equal(report.colors_found, [True, True, True])
    npt.assert_equal(report.objects, 3)

    # Test three points with one color and opacity
    points = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    colors = (0, 1, 0)
    dot_actor = actor.dot(points, colors=colors, opacity=0.8)

    scene.clear()
    scene.add(dot_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    expected_colors = np.floor(0.8 * 255) * np.array([colors])
    report = window.analyze_snapshot(arr, colors=expected_colors)
    npt.assert_equal(report.colors_found, [True])
    npt.assert_equal(report.objects, 3)

    # Test one point with no specified color
    points = np.array([[1, 0, 0]])
    dot_actor = actor.dot(points)

    scene.clear()
    scene.add(dot_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    expected_colors = np.array([[1, 1, 1]]) * 255
    report = window.analyze_snapshot(arr, colors=expected_colors)
    npt.assert_equal(report.colors_found, [True])
    npt.assert_equal(report.objects, 1)


def test_points(interactive=False):
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    opacity = 0.5

    points_actor = actor.point(points, colors, opacity=opacity)

    scene = window.Scene()
    scene.add(points_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)
    npt.assert_equal(points_actor.GetProperty().GetOpacity(), opacity)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 3)


def test_vector_text(interactive=False):
    npt.assert_raises(ExpiredDeprecationError, actor.label, 'FURY Rocks')
    text_actor = actor.vector_text('FURY Rocks', direction=None)

    scene = window.Scene()
    scene.add(text_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    assert text_actor.GetCamera() is scene.GetActiveCamera()

    if interactive:
        window.show(scene, reset_camera=False)

    text_actor = actor.vector_text('FURY Rocks')
    npt.assert_equal(scene.GetActors().GetNumberOfItems(), 1)
    center = np.array(text_actor.GetCenter())
    [assert_greater_equal(v, 0) for v in center]

    text_actor_centered = actor.vector_text('FURY Rocks', align_center=True)
    center = np.array(text_actor_centered.GetCenter())
    npt.assert_equal(center, np.zeros(3))

    text_actor_rot_1 = actor.vector_text('FURY Rocks', direction=(1, 1, 1))
    text_actor_rot_2 = actor.vector_text('FURY Rocks', direction=(1, 1, 0))
    center_1 = text_actor_rot_1.GetCenter()
    center_2 = text_actor_rot_2.GetCenter()
    assert_not_equal(np.linalg.norm(center_1), np.linalg.norm(center_2))

    # test centered
    text_centered = actor.vector_text('FURY Rocks', align_center=True)

    center_3 = text_centered.GetCenter()
    npt.assert_almost_equal(np.linalg.norm(center_3), 0.0)

    text_extruded = actor.vector_text(
        'FURY Rocks', scale=(0.2, 0.2, 0.2), extrusion=1.123
    )
    z_max = text_extruded.GetBounds()[-1]
    npt.assert_almost_equal(z_max, 1.123)

    text_extruded_centered = actor.vector_text(
        'FURY Rocks',
        scale=(0.2, 0.2, 0.2),
        direction=None,
        align_center=True,
        extrusion=23,
    )

    z_min, z_max = text_extruded_centered.GetBounds()[4:]
    npt.assert_almost_equal(z_max - z_min, 23)
    npt.assert_almost_equal(z_max, -z_min)
    # if following the camera, it should rotate around the center to prevent
    # weirdness of the geometry.
    center = np.array(text_actor_centered.GetCenter())
    npt.assert_equal(center, np.zeros(3))


def test_spheres(interactive=False):
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1.0, 0.99]])
    opacity = 0.5

    scene = window.Scene()
    sphere_actor = actor.sphere(
        centers=xyzr[:, :3],
        colors=colors[:],
        radii=xyzr[:, 3],
        opacity=opacity,
        use_primitive=False,
    )
    scene.add(sphere_actor)

    if interactive:
        window.show(scene, order_transparent=True)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 3)
    npt.assert_equal(sphere_actor.GetProperty().GetOpacity(), opacity)

    # test with an unique color for all centers
    scene.clear()
    sphere_actor = actor.sphere(
        centers=xyzr[:, :3],
        colors=np.array([1, 0, 0]),
        radii=xyzr[:, 3],
        use_primitive=False,
    )
    scene.add(sphere_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=(1, 0, 0))
    npt.assert_equal(report.colors_found, [True])

    # test faces and vertices
    scene.clear()
    vertices, faces = fp.prim_sphere(name='symmetric362', gen_faces=False)
    sphere_actor = actor.sphere(
        centers=xyzr[:, :3],
        colors=colors[:],
        radii=xyzr[:, 3],
        opacity=opacity,
        vertices=vertices,
        faces=faces,
    )
    scene.add(sphere_actor)
    if interactive:
        window.show(scene, order_transparent=True)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 3)

    # test primitive sphere type
    scene.clear()
    phi, theta = (30, 30)
    sphere_actor = actor.sphere(
        centers=xyzr[:, :3],
        colors=colors[:],
        radii=xyzr[:, 3],
        opacity=opacity,
        phi=phi,
        theta=theta,
        use_primitive=True,
    )
    scene.add(sphere_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 3)
    npt.assert_equal(sphere_actor.GetProperty().GetOpacity(), opacity)

    # test with unique colors for all spheres
    scene.clear()
    sphere_actor = actor.sphere(
        centers=xyzr[:, :3],
        colors=np.array([0, 0, 1]),
        radii=xyzr[:, 3],
        phi=phi,
        theta=theta,
    )
    scene.add(sphere_actor)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=(0, 0, 255))
    npt.assert_equal(report.colors_found, [True])
    scene.clear()


def test_cones_vertices_faces(interactive=False):
    scene = window.Scene()
    centers = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0], [60, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1, 0.99], [1, 1, 1, 0.6]])
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
    )
    faces = np.array([[0, 1, 3], [0, 2, 1]])
    cone_1 = actor.cone(
        centers=centers[:2],
        directions=directions[:2],
        colors=colors[:2],
        vertices=vertices,
        faces=faces,
        use_primitive=False,
    )

    cone_2 = actor.cone(
        centers=centers[2:],
        directions=directions[2:],
        colors=colors[2:],
        heights=10,
        use_primitive=False,
    )
    scene.add(cone_1)
    scene.add(cone_2)

    if interactive:
        window.show(scene, order_transparent=True)
    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)
    scene.clear()

    # tests for primitive cone
    cone_1 = actor.cone(
        centers=centers[:2],
        directions=directions[:2],
        colors=colors[:2],
        vertices=vertices,
        faces=faces,
    )

    cone_2 = actor.cone(
        centers=centers[2:], directions=directions[2:], colors=colors[2:], heights=10
    )
    scene.add(cone_1)
    scene.add(cone_2)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)
    scene.clear()


def test_basic_geometry_actor(interactive=False):
    centers = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.8], [0, 0, 1, 0.5]])
    directions = np.array([[1, 1, 0]])
    scale_list = [
        1,
        2,
        (1, 1, 1),
        [3, 2, 1],
        np.array([1, 2, 3]),
        np.array([[1, 2, 3], [1, 3, 2], [3, 1, 2]]),
    ]

    actor_list = [
        [actor.cube, {}],
        [actor.box, {}],
        [actor.square, {}],
        [actor.rectangle, {}],
        [actor.frustum, {}],
        [actor.octagonalprism, {}],
        [actor.pentagonalprism, {}],
        [actor.triangularprism, {}],
        [actor.rhombicuboctahedron, {}],
    ]

    for act_func, extra_args in actor_list:
        for scale in scale_list:
            scene = window.Scene()
            g_actor = act_func(
                centers=centers,
                colors=colors,
                directions=directions,
                scales=scale,
                **extra_args,
            )

            scene.add(g_actor)
            if interactive:
                window.show(scene)

            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr, colors=colors)
            msg = 'Failed with {}, scale={}'.format(act_func.__name__, scale)
            npt.assert_equal(report.objects, 3, err_msg=msg)


def test_advanced_geometry_actor(interactive=False):
    xyz = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    dirs = np.array([[0.5, 0.5, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]])

    heights = np.array([5, 7, 10])

    actor_list = [
        [actor.cone, {'heights': heights, 'resolution': 8}],
        [actor.arrow, {'scales': heights, 'resolution': 9}],
        [actor.cylinder, {'heights': heights, 'resolution': 10}],
        [actor.disk, {'rinner': 4, 'router': 8, 'rresolution': 2, 'cresolution': 2}],
    ]

    scene = window.Scene()

    for act_func, extra_args in actor_list:
        colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [1, 1, 0, 1]])

        geom_actor = act_func(xyz, dirs, colors, **extra_args)
        scene.add(geom_actor)

        if interactive:
            window.show(scene, order_transparent=True)
        arr = window.snapshot(scene)
        report = window.analyze_snapshot(arr, colors=colors)
        npt.assert_equal(report.objects, 3)

        scene.clear()

        colors = np.array([1.0, 1.0, 1.0, 1.0])

        geom_actor = act_func(xyz, dirs, colors, **extra_args)
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
    # X axis of right alignment should have a higher center of mass position
    # than left
    assert_greater(center_of_mass(arr_right)[0], center_of_mass(arr_left)[0])
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
    assert_greater_equal(center_of_mass(arr_top)[0], center_of_mass(arr_bottom)[0])

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
    npt.assert_equal(container_shallow_copy.GetPosition(), container.GetPosition())
    npt.assert_equal(container_shallow_copy.GetVisibility(), container.GetVisibility())

    # Check is the shallow_copy do not modify original container
    container_shallow_copy.SetVisibility(False)
    npt.assert_equal(container.GetVisibility(), True)

    container_shallow_copy.SetPosition((1, 1, 1))
    npt.assert_equal(container.GetPosition(), (0, 0, 0))


def test_grid(_interactive=False):
    vol1 = np.zeros((100, 100, 100))
    vol1[25:75, 25:75, 25:75] = 100
    contour_actor1 = actor.contour_from_roi(vol1, np.eye(4), (1.0, 0, 0), 1.0)

    vol2 = np.zeros((100, 100, 100))
    vol2[25:75, 25:75, 25:75] = 100

    contour_actor2 = actor.contour_from_roi(vol2, np.eye(4), (1.0, 0.5, 0), 1.0)
    vol3 = np.zeros((100, 100, 100))
    vol3[25:75, 25:75, 25:75] = 100

    contour_actor3 = actor.contour_from_roi(vol3, np.eye(4), (1.0, 0.5, 0.5), 1.0)

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
    container = grid(
        actors=actors,
        captions=None,
        caption_offset=(0, -40, 0),
        cell_padding=(10, 10),
        dim=(2, 3),
    )

    scene.add(container)

    scene.projection('orthogonal')

    counter = itertools.count()

    show_m = window.ShowManager(scene)

    def timer_callback(_obj, _event):
        nonlocal counter
        cnt = next(counter)
        # show_m.scene.zoom(1)
        show_m.render()
        if cnt == 5:
            show_m.exit()
            # show_m.destroy_timers()

    show_m.add_timer_callback(True, 200, timer_callback)
    show_m.start()

    arr = window.snapshot(scene)
    arr[arr < 100] = 0
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 6)

    scene.rm_all()

    counter = itertools.count()
    show_m = window.ShowManager(scene)

    # show the grid with the captions
    container = grid(
        actors=actors,
        captions=texts,
        caption_offset=(0, -50, 0),
        cell_padding=(10, 10),
        dim=(3, 3),
    )

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

    arr[rs - w : rs + w, cs - 10 * w : cs + 10 * w] = np.array([255, 127, 0])
    # enable to see pacman on sphere
    # arr[0: 2 * w, cs - 10 * w: cs + 10 * w] = np.array([255, 127, 0])
    scene = window.Scene()
    tsa = actor.texture_on_sphere(arr)
    scene.add(tsa)
    rotate(tsa, rotation=(90, 0, 1, 0))
    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display, bg_color=(0, 0, 0), colors=[(255, 127, 0)], find_objects=False
    )
    npt.assert_equal(res.colors_found, [True])


def test_texture_mapping():
    arr = np.zeros((512, 212, 3), dtype='uint8')
    arr[:256, :] = np.array([255, 0, 0])
    arr[256:, :] = np.array([0, 255, 0])
    tp = actor.texture(arr, interp=True)
    scene = window.Scene()
    scene.add(tp)
    display = window.snapshot(scene)
    res = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(255, 0, 0), (0, 255, 0)],
        find_objects=False,
    )
    npt.assert_equal(res.colors_found, [True, True])


def test_texture_update():
    arr = np.zeros((512, 212, 3), dtype='uint8')
    arr[:256, :] = np.array([255, 0, 0])
    arr[256:, :] = np.array([0, 255, 0])
    # create a texture on plane
    tp = actor.texture(arr, interp=True)
    scene = window.Scene()
    scene.add(tp)
    display = window.snapshot(scene)
    res1 = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
        find_objects=False,
    )

    # update the texture
    new_arr = np.zeros((512, 212, 3), dtype='uint8')
    new_arr[:, :] = np.array([255, 255, 255])
    actor.texture_update(tp, new_arr)
    display = window.snapshot(scene)
    res2 = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
        find_objects=False,
    )

    # Test for original colors
    npt.assert_equal(res1.colors_found, [False, True, True])
    # Test for changed colors of the actor
    npt.assert_equal(res2.colors_found, [True, False, False])


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
    res = window.analyze_snapshot(
        display,
        bg_color=(0, 0, 0),
        colors=[(255, 0, 0), (255, 255, 255)],
        find_objects=False,
    )
    npt.assert_equal(res.colors_found, [True, True])


@pytest.mark.skipif(not have_matplotlib, reason='Requires MatplotLib')
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
    plt.close('all')
    fig_actor = actor.figure(arr, 'cubic')
    fig_actor2 = actor.figure(arr, 'cubic')
    scene = window.Scene()
    scene.background((1, 1, 1.0))

    ax_actor = actor.axes(scale=(1000, 1000, 1000))
    scene.add(ax_actor)
    scene.add(fig_actor)
    scene.add(fig_actor2)
    ax_actor.SetPosition(-50, 500, -800)
    fig_actor2.SetPosition(500, 800, -400)
    display = window.snapshot(
        scene, 'test_mpl.png', order_transparent=False, offscreen=True
    )
    res = window.analyze_snapshot(
        display, bg_color=(255, 255, 255.0), colors=[(31, 119, 180)], find_objects=False
    )
    # omit assert from now until we know why snapshot creates
    # different colors in Github Actions but not on our computers
    # npt.assert_equal(res.colors_found, [True, True])
    # TODO: investigate further this issue with snapshot in Actions
    pass


def test_superquadric_actor(interactive=False):
    scene = window.Scene()
    centers = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    directions = np.array(
        [
            [0.27753247, 0.15332503, 0.63670953],
            [0.14138223, 0.76031677, 0.14669451],
            [0.23416946, 0.12816617, 0.92596145],
        ]
    )

    scales = [1, 2, 3]
    roundness = np.array([[1, 1], [1, 2], [2, 1]])

    sq_actor = actor.superquadric(
        centers,
        roundness=roundness,
        directions=directions,
        colors=colors.astype(np.uint8),
        scales=scales,
    )
    scene.add(sq_actor)
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene, offscreen=True)
    arr[arr > 0] = 255  # Normalization
    report = window.analyze_snapshot(arr, colors=255 * colors.astype(np.uint8))
    npt.assert_equal(report.objects, 3)
    npt.assert_equal(report.colors_found, [True, True, True])


def test_billboard_actor(interactive=False):
    scene = window.Scene()
    scene.background((1, 1, 1))
    centers = np.array(
        [
            [0, 0, 0],
            [5, -5, 5],
            [-7, 7, -7],
            [10, 10, 10],
            [10.5, 11.5, 11.5],
            [12, -12, -12],
            [-17, 17, 17],
            [-22, -22, 22],
        ]
    )
    colors = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )
    scales = [6, 0.4, 1.2, 1, 0.2, 0.7, 3, 2]

    fake_sphere = """
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

    billboard_actor = actor.billboard(
        centers, colors=colors, scales=scales, fs_impl=fake_sphere
    )
    scene.add(billboard_actor)

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 8)
    scene.clear()

    centers = np.array([[0, 0, 0], [-15, 15, -5], [10, -10, 5],
                        [-30, 30, -10], [20, -20, 10]])
    colors = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1],
                       [1, 0, 0], [0, 1, 0]])
    scales = [3, 1, 2, 1, 1.5]

    b_point = """
        float len = length(point);
        float radius = .2;
        if(len > radius)
            {fragOutput0 = vec4(vec3(0,0,0), 1);}
        else
            {fragOutput0 = vec4(color, 1);}
        """

    b_type = ['spherical', 'cylindrical_x', 'cylindrical_y']
    expected_val = [True, False, False]
    rotations = [[87, 0, -87, 87], [87, 0, -87, 87], [0, 87, 87, -87]]
    for i in range(3):
        billboard = actor.billboard(centers, colors=colors, scales=scales,
                                    bb_type=b_type[i], fs_impl=b_point)

        scene.add(billboard)
        if b_type[i] == 'spherical':
            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr, colors=255 * colors)
            npt.assert_equal(report.colors_found, [True] * 5)

        scene.pitch(rotations[i][0])
        scene.yaw(rotations[i][1])
        if interactive:
            window.show(scene)

        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene, offscreen=True)
        report = window.analyze_snapshot(arr, colors=255 * colors)
        npt.assert_equal(report.colors_found, [True] * 5)

        scene.pitch(rotations[i][2])
        scene.yaw(rotations[i][3])
        if interactive:
            window.show(scene)

        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene, offscreen=True)
        report = window.analyze_snapshot(arr, colors=255 * colors)
        npt.assert_equal(report.colors_found, [expected_val[i]] * 5)

        scene.yaw(-87)
        scene.clear()


@pytest.mark.skipif(
    skip_win,
    reason='This test does not work on Windows'
    ' due to snapshot (memory access'
    ' violation). Check what is causing this'
    ' issue with shader',
)
def test_sdf_actor(interactive=False):
    scene = window.Scene()
    scene.background((1, 1, 1))
    centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0], [2, 2, 0]]) * 11
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]])
    scales = [1, 2, 3, 4]
    primitive = ['sphere', 'ellipsoid', 'torus', 'capsule']

    sdf_actor = actor.sdf(centers, directions, colors, primitive, scales)
    scene.add(sdf_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)

    # Draw 3 spheres as the primitive type is str
    scene.clear()
    primitive = 'sphere'
    sdf_actor = actor.sdf(centers, directions, colors, primitive, scales)
    scene.add(sdf_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)

    # A sphere and default back to two torus
    # as the primitive type is list
    scene.clear()
    primitive = ['sphere']
    with npt.assert_warns(UserWarning):
        sdf_actor = actor.sdf(centers, directions, colors, primitive, scales)

    scene.add(sdf_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)

    # One sphere and ellipsoid each
    # Default to torus
    scene.clear()
    primitive = ['sphere', 'ellipsoid']
    with npt.assert_warns(UserWarning):
        sdf_actor = actor.sdf(centers, directions, colors, primitive, scales)

    scene.add(sdf_actor)
    scene.add(actor.axes())
    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)


@pytest.mark.skipif(
    skip_linux,
    reason='This test does not work on Ubuntu. It '
    'works on a local machine. Check after '
    'fixing memory leak with RenderWindow.',
)
def test_marker_actor(interactive=False):
    scene = window.Scene()
    scene.background((1, 1, 1))
    centers_3do = np.array([[4, 0, 0], [4, 4, 0], [4, 8, 0]])
    markers_2d = ['o', 's', 'd', '^', 'p', 'h', 's6', 'x', '+']
    center_markers_2d = np.array([[0, i * 2, 0] for i in range(len(markers_2d))])
    fake_spheres = actor.markers(centers_3do, colors=(0, 1, 0), scales=1, marker='3d')
    markers_2d = actor.markers(
        center_markers_2d, colors=(0, 1, 0), scales=1, marker=markers_2d
    )
    scene.add(fake_spheres)
    scene.add(markers_2d)

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)

    colors = np.array([[0, 1, 0] for i in range(12)])
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 12)


def test_ellipsoid_actor(interactive=False):
    # number of axes does not match with number of centers
    centers = [-1, 1, 0]
    axes = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 2, -2], [2, 1, 2], [2, -2, -1]]]
    lengths = [[1, 1, 1]]
    npt.assert_raises(ValueError, actor.ellipsoid, centers, axes, lengths)

    # number of lengths does not match with number of centers
    lengths = [[1, 1, 1], [1, 1, .5]]
    npt.assert_raises(ValueError, actor.ellipsoid, centers, axes, lengths)

    scene = window.Scene()
    scene.background((0, 0, 0))

    axes = np.array([[[-.6, .5, -.6], [-.8, -.4, .5], [-.1, -.7, -.7]],
                     [[.1, .6, -.8], [.6, .5, .5], [-.8, .6, .3]],
                     [[.7, .5, -.5], [0, -.7, -.7], [-.7, .6, -.5]],
                     [[.7, -.3, -.6], [.2, -.8, .6], [.7, .6, .5]],
                     [[1, 2, -2], [2, 1, 2], [2, -2, -1]],
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    lengths = np.array([[1, 1, 1], [1, 1, .5], [1, .5, .5],
                        [1, .5, .25], [1, 1, .3], [1, .3, .3]])
    centers = np.array([[-1, 1, 0], [0, 1, 0], [1, 1, 0],
                        [-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 0], [1, 0, 1], [0, 1, 1]])

    ellipsoids = actor.ellipsoid(axes=axes, lengths=lengths, centers=centers,
                              scales=1.0, colors=colors)
    scene.add(ellipsoids)

    if interactive:
        window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors, 1)


def test_uncertainty_cone_actor(interactive=False):
    scene = window.Scene()

    evals = np.array([1.4, 0.5, 0.35])
    evecs = np.eye(3)

    mevals = np.zeros((10, 10, 1, 3))
    mevecs = np.zeros((10, 10, 1, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    signal = np.ones((10, 10, 1, 10))
    sigma = np.array([14.791911, 14.999622, 14.880976, 14.933881, 14.392784,
                      14.132468, 14.334953, 14.409375, 14.514647, 14.409275])

    b_matrix = np.array([[-1.8, -1.9, -4.8, -4.4, -2.3, -1.2, -1.0],
                         [-5.4, -1.8, -1.6, -1.7, -6.1, -1.3, -1.0],
                         [-6.2, -5.1, -1.0, -1.9, -9.3, -2.2, -1.0],
                         [-2.8, -1.9, -4.8, -1.4, -2.1, -3.6, -1.0],
                         [-5.6, -1.3, -7.8, -2.4, -5.2, -4.2, -1.0],
                         [-1.8, -2.5, -1.8, -1.2, -2.3, -4.8, -1.0],
                         [-2.3, -1.9, -6.8, -4.4, -6.4, -1.9, -1.0],
                         [-1.8, -2.6, -4.8, -6.5, -7.7, -3.1, -1.0],
                         [-6.2, -1.9, -5.6, -4.6, -1.5, -2.0, -1.0],
                         [-2.4, -1.9, -4.5, -3.6, -2.5, -1.2, -1.0]])

    uncert_cones = actor.uncertainty_cone(evecs=mevecs, evals=mevals,
                                          signal=signal, sigma=sigma,
                                          b_matrix=b_matrix)
    scene.add(uncert_cones)

    if interactive:
        window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors, 1)
    scene.clear()

    evals = np.array([1.4, 0.5, 0.35])
    evecs = np.eye(3)

    mevals = np.zeros((4, 4, 4, 3))
    mevecs = np.zeros((4, 4, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    signal = np.ones((4, 4, 4, 10))
    uncert_cones = actor.uncertainty_cone(evecs=mevecs, evals=mevals,
                                          signal=signal, sigma=sigma,
                                          b_matrix=b_matrix)
    scene.add(uncert_cones)

    if interactive:
        window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors, 1)
    scene.clear()


def test_actors_primitives_count():
    centers = np.array([[1, 1, 1], [2, 2, 2]])
    directions = np.array([[1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [1, 0, 0]])
    lines = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]])

    args_1 = {'centers': centers}
    args_2 = {**args_1, 'colors': colors}
    args_3 = {**args_2, 'directions': directions}

    cen_c = len(centers)
    lin_c = len(lines)

    actors_test_cases = [
        [actor.box, args_1, cen_c],
        [actor.rectangle, args_1, cen_c],
        [actor.square, args_1, cen_c],
        [actor.cube, args_1, cen_c],
        [actor.sphere, {**args_2, 'use_primitive': True}, cen_c],
        [actor.sphere, {**args_2, 'use_primitive': False}, cen_c],
        [actor.sdf, args_1, cen_c],
        [actor.billboard, args_1, cen_c],
        [actor.superquadric, args_1, cen_c],
        [actor.markers, args_1, cen_c],
        [actor.octagonalprism, args_1, cen_c],
        [actor.frustum, args_1, cen_c],
        [actor.pentagonalprism, args_1, cen_c],
        [actor.triangularprism, args_1, cen_c],
        [actor.rhombicuboctahedron, args_1, cen_c],
        [actor.cylinder, args_3, cen_c],
        [actor.disk, args_3, cen_c],
        [actor.cone, {**args_3, 'use_primitive': False}, cen_c],
        [actor.cone, {**args_3, 'use_primitive': True}, cen_c],
        [actor.arrow, {**args_3, 'repeat_primitive': False}, cen_c],
        [actor.arrow, {**args_3, 'repeat_primitive': True}, cen_c],
        [actor.dot, {'points': centers}, cen_c],
        [actor.point, {'points': centers, 'colors': colors}, cen_c],
        [actor.line, {'lines': lines}, lin_c],
        [actor.streamtube, {'lines': lines}, lin_c],
    ]
    for test_case in actors_test_cases:
        act_func = test_case[0]
        args = test_case[1]
        primitives_count = test_case[2]
        act = act_func(**args)
        npt.assert_equal(primitives_count_from_actor(act), primitives_count)
