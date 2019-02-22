"""Function for testing colormap module."""

import numpy as np
import numpy.testing as npt

from fury import colormap


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

    cmap = colormap.get_cmap('Accent')
    npt.assert_array_almost_equal(cmap((1, 0, 0)), expected2)
