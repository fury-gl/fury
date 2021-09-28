from fury import actor, window
from fury.actors.peak import (PeakActor, _orientation_colors,
                              _peaks_colors_from_points, _points_to_vtk_cells)
from fury.lib import numpy_support


import numpy as np
import numpy.testing as npt


def generate_peaks():
    dirs01 = np.array([[-.4, .4, .8], [.7, .6, .1], [.4, -.3, .2],
                       [0, 0, 0], [0, 0, 0]])
    dirs10 = np.array([[.6, -.6, -.2], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])
    dirs11 = np.array([[0., .3, .3], [-.8, .4, -.5], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])
    dirs12 = np.array([[0, 0, 0], [.7, .6, .1], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])

    peaks_dirs = np.zeros((1, 2, 3, 5, 3))

    peaks_dirs[0, 0, 1, :, :] = dirs01
    peaks_dirs[0, 1, 0, :, :] = dirs10
    peaks_dirs[0, 1, 1, :, :] = dirs11
    peaks_dirs[0, 1, 2, :, :] = dirs12

    peaks_vals = np.zeros((1, 2, 3, 5))

    peaks_vals[0, 0, 1, :] = np.array([.3, .2, .6, 0, 0])
    peaks_vals[0, 1, 0, :] = np.array([.5, 0, 0, 0, 0])
    peaks_vals[0, 1, 1, :] = np.array([.2, .5, 0, 0, 0])
    peaks_vals[0, 1, 2, :] = np.array([0, .7, 0, 0, 0])
    return peaks_dirs, peaks_vals, np.eye(4)


def test__orientation_colors():
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1]])

    colors = _orientation_colors(points, cmap='rgb_standard')
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    npt.assert_array_equal(colors, expected)

    npt.assert_raises(ValueError, _orientation_colors, points, cmap='test')


def test__peaks_colors_from_points():
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1]])

    colors_tuple = _peaks_colors_from_points(points, colors=None)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
               [0, 0, 0]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, 1)

    colors_tuple = _peaks_colors_from_points(points, colors='rgb_standard')
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
               [0, 0, 0]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, 1)

    colors_tuple = _peaks_colors_from_points(points, colors=(0, 1, 0))
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, 1)

    colors_tuple = _peaks_colors_from_points(points, colors=(0, 1, 0, .1))
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[0, 255, 0, 25], [0, 255, 0, 25], [0, 255, 0, 25],
               [0, 255, 0, 25], [0, 255, 0, 25], [0, 255, 0, 25]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, -1)

    colors = [.3, .6, 1]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [.3, .3, .6, .6, 1, 1]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, True)
    npt.assert_equal(global_opacity, 1)

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0],
               [0, 0, 255], [0, 0, 255]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, 1)

    colors = [[1, 0, 0, .1], [0, 1, 0, .1], [0, 0, 1, .1]]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[255, 0, 0, 25], [255, 0, 0, 25], [0, 255, 0, 25],
               [0, 255, 0, 25], [0, 0, 255, 25], [0, 0, 255, 25]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, -1)

    colors = [.5, .6, .7, .8, .9, 1]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [.5, .6, .7, .8, .9, 1]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, True)
    npt.assert_equal(global_opacity, 1)

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
               [255, 0, 255], [0, 255, 255]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, 1)

    colors = [[1, 0, 0, .1], [0, 1, 0, .1], [0, 0, 1, .1], [1, 1, 0, .1],
              [1, 0, 1, .1], [0, 1, 1, .1]]
    colors_tuple = _peaks_colors_from_points(points, colors=colors)
    vtk_colors, colors_are_scalars, global_opacity = colors_tuple
    desired = [[255, 0, 0, 25], [0, 255, 0, 25], [0, 0, 255, 25],
               [255, 255, 0, 25], [255, 0, 255, 25], [0, 255, 255, 25]]
    npt.assert_array_equal(numpy_support.vtk_to_numpy(vtk_colors), desired)
    npt.assert_equal(colors_are_scalars, False)
    npt.assert_equal(global_opacity, -1)


def test__points_to_vtk_cells():
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1]])

    vtk_cells = _points_to_vtk_cells(points)
    actual = numpy_support.vtk_to_numpy(vtk_cells.GetData())
    desired = [2, 0, 1, 2, 2, 3, 2, 4, 5]
    npt.assert_array_equal(actual, desired)
    npt.assert_equal(vtk_cells.GetNumberOfCells(), 3)


def test_colors(interactive=False):
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    scene = window.Scene()

    colors = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
              [1, 1, 0]]

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine, colors=colors)

    scene.add(peak_actor)

    scene.azimuth(30)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 4)


def test_display_cross_section():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    peak_actor.display_cross_section(0, 0, 0)
    npt.assert_equal(peak_actor.is_range, False)
    npt.assert_equal(peak_actor.cross_section, [0, 0, 0])

    peak_actor.display_extent(0, 0, 0, 1, 0, 2)
    npt.assert_equal(peak_actor.is_range, True)

    peak_actor.display_cross_section(0, 0, 0)
    npt.assert_equal(peak_actor.is_range, False)


def test_display_extent():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    peak_actor.display_extent(0, 0, 0, 1, 0, 2)
    npt.assert_equal(peak_actor.is_range, True)
    npt.assert_equal(peak_actor.low_ranges, [0, 0, 0])
    npt.assert_equal(peak_actor.high_ranges, [0, 1, 2])

    peak_actor.display_cross_section(0, 0, 0)
    npt.assert_equal(peak_actor.is_range, False)

    peak_actor.display_extent(0, 0, 0, 1, 0, 2)
    npt.assert_equal(peak_actor.is_range, True)


def test_global_opacity():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    npt.assert_equal(peak_actor.global_opacity, 1)

    peak_actor.global_opacity = .5
    npt.assert_equal(peak_actor.global_opacity, .5)

    peak_actor.global_opacity = 0
    npt.assert_equal(peak_actor.global_opacity, 0)


def test_linewidth():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    npt.assert_equal(peak_actor.linewidth, 1)

    peak_actor.linewidth = 2
    npt.assert_equal(peak_actor.linewidth, 2)

    peak_actor.linewidth = 5
    npt.assert_equal(peak_actor.linewidth, 5)

    peak_actor.linewidth = 0
    npt.assert_equal(peak_actor.linewidth, 0)


def test_lookup_colormap(interactive=False):
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    scene = window.Scene()

    colors = [.0, .1, .2, .5, .8, .9, 1]

    hue = (0, 1)  # Red to blue
    saturation = (0, 1)  # White to full saturation

    lut_cmap = actor.colormap_lookup_table(
        hue_range=hue, saturation_range=saturation)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine, colors=colors,
                           lookup_colormap=lut_cmap)

    scene.add(peak_actor)

    scene.azimuth(30)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 4)


def test_max_centers():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    npt.assert_equal(peak_actor.max_centers, [0, 1, 2])


def test_min_centers():
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0
    indices = np.nonzero(valid_mask)

    peak_actor = PeakActor(peak_dirs, indices, values=peak_vals,
                           affine=peak_affine)

    npt.assert_equal(peak_actor.min_centers, [0, 0, 0])
