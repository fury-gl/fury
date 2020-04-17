"""Function for testing colormap module."""

import numpy as np
import numpy.testing as npt

from fury import colormap
from fury.optpkg import optional_package
cm, have_matplotlib, _ = optional_package('matplotlib.cm')


def test_boys2rgb():
    expected = np.array([[0.23171663, 0.34383397, 0.6950296],
                         [0.74520645, 0.58600913, 0.6950296],
                         [0.48846154, 0.46492155, 0.05164146]]
                        )
    v1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v2 = np.array([1, 0, 0, 0, 1, 0])

    for v, e in zip([v1, v2], [expected, expected[0]]):
        c = colormap.boys2rgb(v)
        npt.assert_array_almost_equal(c, e)


def test_orient2rgb():
    e2 = [0.70710678, 0, 0, 0, 0.70710678, 0]
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v2 = np.array([1, 0, 0, 0, 1, 0])
    npt.assert_equal(colormap.orient2rgb(v), v)
    npt.assert_almost_equal(colormap.orient2rgb(v2), e2)
    npt.assert_raises(IOError, colormap.orient2rgb, np.array(1))


def test_get_cmap():
    npt.assert_equal(colormap.get_cmap(''), None)
    npt.assert_equal(colormap.get_cmap('blues'), None)

    expected = np.array([[0.03137255, 0.1882353, 0.41960785, 1],
                         [0.96862745, 0.98431373, 1, 1],
                         [0.96862745, 0.98431373, 1, 1]]
                        )
    expected2 = np.array([[0.4, 0.4, 0.4, 1.],
                          [0.498039, 0.788235, 0.498039, 1],
                          [0.498039, 0.788235, 0.498039, 1]])
    cmap = colormap.get_cmap('Blues')
    npt.assert_array_almost_equal(cmap((1, 0, 0)), expected)

    with npt.assert_warns(PendingDeprecationWarning):
        cmap = colormap.get_cmap('Accent')
    npt.assert_array_almost_equal(cmap((1, 0, 0)), expected2)


def test_line_colors():
    s1 = np.array([np.arange(10)]*3).T  # 10x3
    s2 = np.array([np.arange(5)]*4)  # 5x4
    streamlines = [s1, s2]

    s_color = colormap.line_colors(streamlines, cmap='boys_standard')
    npt.assert_equal(s_color.shape, (2, 3))


def test_create_colormap():
    value = np.arange(25)
    npt.assert_raises(ValueError, colormap.create_colormap,
                      value.reshape((5, 5)))
    npt.assert_raises(ValueError, colormap.create_colormap,
                      value, name='fake')
    npt.assert_warns(PendingDeprecationWarning, colormap.create_colormap,
                     value, name='jet', auto=False)

    if not have_matplotlib:
        with npt.assert_warns(UserWarning):
            npt.assert_raises(ValueError, colormap.create_colormap, value)


def test_lab_delta():
    color = np.c_[100, 127, 128]
    delta = np.c_[0, 0, 0]

    res = colormap._lab_delta(color, color)
    res_2 = colormap._lab_delta(color, delta)
    npt.assert_equal(res, 0)
    npt.assert_equal(np.round(res_2), [206])


def test_rgb_lab_delta():
    color = np.c_[255, 65, 0]
    delta = np.c_[0, 65, 0]

    res = colormap._rgb_lab_delta(color, color)
    npt.assert_equal(res, 0)
    res = colormap._rgb_lab_delta(color, delta)
    npt.assert_equal(np.round(res), [114])


def test_lab2xyz():
    lab_color = np.c_[100, 128, 128]
    expected = np.c_[188.32, 100, 5.08]

    res = colormap._lab2xyz(lab_color)
    npt.assert_array_almost_equal(res, expected, decimal=2)


def test_xyz2rgb():
    xyz_color = np.c_[43.14, 25.07, 2.56]
    expected = np.c_[255, 65, 0]

    res = np.round(colormap._xyz2rgb(xyz_color))
    npt.assert_array_almost_equal(res, expected)


def test_lab2rgb():
    lab_color = np.c_[0, 128, 128]
    expected = np.c_[133, 0, 0]

    res = np.round(colormap._lab2rgb(lab_color))
    res[res < 0] = 0
    npt.assert_array_almost_equal(res, expected)


def test_hex_to_rgb():
    expected = np.array([1, 1, 1])
    
    hexcode = "#FFFFFF"
    res = colormap.hex_to_rgb(hexcode)
    npt.assert_array_almost_equal(res, expected)
    
    hashed_hexcode = "FFFFFF"
    res = colormap.hex_to_rgb(hashed_hexcode)
    npt.assert_array_almost_equal(res, expected)
