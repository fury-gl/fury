"""Function for testing colormap module."""

import numpy as np
import numpy.testing as npt
import pytest

from fury import colormap
from fury.optpkg import optional_package

cm, have_matplotlib, _ = optional_package("matplotlib.cm")


def test_boys2rgb():
    expected = np.array(
        [
            [0.23171663, 0.34383397, 0.6950296],
            [0.74520645, 0.58600913, 0.6950296],
            [0.48846154, 0.46492155, 0.05164146],
        ]
    )
    v1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v2 = np.array([1, 0, 0, 0, 1, 0])

    for v, e in zip([v1, v2], [expected, expected[0]], strict=False):
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
    npt.assert_equal(colormap.get_cmap(""), None)
    npt.assert_equal(colormap.get_cmap("blues"), None)

    expected = np.array(
        [
            [0.03137255, 0.1882353, 0.41960785, 1],
            [0.96862745, 0.98431373, 1, 1],
            [0.96862745, 0.98431373, 1, 1],
        ]
    )
    expected2 = np.array(
        [
            [0.4, 0.4, 0.4, 1.0],
            [0.498039, 0.788235, 0.498039, 1],
            [0.498039, 0.788235, 0.498039, 1],
        ]
    )
    cmap = colormap.get_cmap("Blues")
    npt.assert_array_almost_equal(cmap((1, 0, 0)), expected)

    with pytest.warns(PendingDeprecationWarning):
        cmap = colormap.get_cmap("Accent")
    npt.assert_array_almost_equal(cmap((1, 0, 0)), expected2)


def test_line_colors():
    s1 = np.array([np.arange(10)] * 3).T  # 10x3
    s2 = np.array([np.arange(5)] * 4)  # 5x4
    streamlines = [s1, s2]

    s_color = colormap.line_colors(streamlines, cmap="boys_standard")
    npt.assert_equal(s_color.shape, (2, 3))


def test_create_colormap():
    value = np.arange(25)
    npt.assert_raises(
        ValueError,
        colormap.create_colormap,
        value.reshape((5, 5)),
        name="plasma",
        auto=True,
    )
    npt.assert_raises(
        AttributeError if have_matplotlib else ValueError,
        colormap.create_colormap,
        value,
        name="fake",
        auto=True,
    )
    with pytest.warns(PendingDeprecationWarning):
        colormap.create_colormap(value, name="jet", auto=True)

    if not have_matplotlib:
        with pytest.warns(UserWarning):
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


def test_color_converters():
    color = np.array([1, 1, 1])
    colors = np.array([[1, 1, 1], [0, 0, 0], [0.2, 0.3, 0.4]])

    # testing rgb2xyz and xyz2rgb
    expected_xyz = np.array([0.950456, 1.0, 1.088754])
    xyz_color = colormap.rgb2xyz(color)
    rgb_color = colormap.xyz2rgb(expected_xyz)
    npt.assert_almost_equal(xyz_color, expected_xyz)
    npt.assert_almost_equal(rgb_color, color)

    for color in colors:
        xyz_color = colormap.rgb2xyz(color)
        rgb_from_xyz_color = colormap.xyz2rgb(xyz_color)
        npt.assert_almost_equal(rgb_from_xyz_color, color)

    # testing rgb2lab and lab2rgb
    illuminant = "D65"
    observer = "2"
    expected_lab = np.array([31.57976662, -1.86550104, -17.84845331])
    lab_color = colormap.rgb2lab(color, illuminant=illuminant, observer=observer)
    rgb_color = colormap.lab2rgb(expected_lab, illuminant=illuminant, observer=observer)
    npt.assert_almost_equal(lab_color, expected_lab)
    npt.assert_almost_equal(rgb_color, color)

    for color in colors:
        lab_color = colormap.rgb2lab(color, illuminant=illuminant, observer=observer)
        rgb_from_lab_color = colormap.lab2rgb(
            lab_color, illuminant=illuminant, observer=observer
        )
        npt.assert_almost_equal(rgb_from_lab_color, color)

    # testing rgb2hsv and hsv2rgb
    expected_hsv = np.array([0.58333333, 0.5, 0.4])
    hsv_color = colormap.rgb2hsv(color)
    rgb_color = colormap.hsv2rgb(expected_hsv)
    npt.assert_almost_equal(hsv_color, expected_hsv)
    npt.assert_almost_equal(rgb_color, color)

    for color in colors:
        hsv_color = colormap.rgb2hsv(color)
        rgb_from_hsv_color = colormap.hsv2rgb(hsv_color)
        npt.assert_almost_equal(rgb_from_hsv_color, color)


def test_normalize_colors():
    from fury.colormap import normalize_colors

    # [0, 255] RGB tuple
    result = normalize_colors((255, 0, 0))
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0]])
    assert result.dtype == np.float32

    # [0, 1] RGB tuple (backward compat)
    result = normalize_colors((0.5, 0.5, 0.5))
    npt.assert_array_almost_equal(result, [[0.5, 0.5, 0.5]])
    assert result.dtype == np.float32

    # [0, 255] ndarray
    colors_255 = np.array([[255, 0, 0], [0, 255, 0]])
    result = normalize_colors(colors_255)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    npt.assert_array_almost_equal(result, expected)
    assert result.shape == (2, 3)

    # RGBA [0, 255]
    result = normalize_colors((255, 0, 0, 128))
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0, 128 / 255.0]])
    assert result.shape == (1, 4)

    # RGBA [0, 1]
    result = normalize_colors((1.0, 0.0, 0.0, 0.5))
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0, 0.5]])

    # Hex string
    result = normalize_colors("#FF0000")
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0]])

    # Hex list
    result = normalize_colors(["#FF0000", "#00FF00"])
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert result.shape == (2, 3)

    # n_points broadcasting
    result = normalize_colors((255, 0, 0), n_points=5)
    assert result.shape == (5, 3)
    npt.assert_array_almost_equal(result[0], [1.0, 0.0, 0.0])
    npt.assert_array_almost_equal(result[4], [1.0, 0.0, 0.0])

    # None → default red
    result = normalize_colors(None)
    npt.assert_array_almost_equal(result, [[1.0, 0.0, 0.0]])

    # Single color with n_points broadcasts (no error)
    result = normalize_colors((255, 0, 0), n_points=3)
    assert result.shape == (3, 3)

    # Multiple colors mismatching n_points → ValueError
    with pytest.raises(ValueError):
        normalize_colors(np.array([[255, 0, 0], [0, 255, 0]]), n_points=5)

    # Output dtype is always float32
    result = normalize_colors(np.array([[0.1, 0.2, 0.3]], dtype=np.float64))
    assert result.dtype == np.float32

    # Black (0, 0, 0) passes through correctly
    result = normalize_colors((0, 0, 0))
    npt.assert_array_almost_equal(result, [[0.0, 0.0, 0.0]])

    # Hex string with n_points broadcasting
    result = normalize_colors("#FF0000", n_points=3)
    assert result.shape == (3, 3)

    # Hex list with n_points mismatch
    with pytest.raises(ValueError):
        normalize_colors(["#FF0000", "#00FF00"], n_points=5)

    # Invalid channel count
    with pytest.raises(ValueError):
        normalize_colors((1, 2))

    with pytest.raises(ValueError):
        normalize_colors((1, 2, 3, 4, 5))
