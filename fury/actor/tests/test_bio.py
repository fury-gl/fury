import numpy as np
import pytest

from fury.actor import (
    Group,
    contour_from_label,
    contour_from_roi,
    get_slices,
    peaks_slicer,
    volume_slicer,
)
from fury.lib import AffineTransform


def test_volume_slicer_comprehensive():
    """Test all major functionality of volume_slicer in one comprehensive test."""
    # Create test data - a simple 10x10x10 volume with gradient values
    data = np.random.rand(10, 10, 10)
    for i in range(10):
        data[i, :, :] = i / 10.0

    # Create test affine matrix with scaling and translation
    affine = np.eye(4)
    affine[0, 0] = 2.0  # Scale x-axis by 2
    affine[1, 1] = 0.5  # Scale y-axis by 0.5
    affine[2, 3] = 5.0  # Translate z-axis by 5

    # Test with all parameters specified
    actor = volume_slicer(
        data,
        affine=affine,
        value_range=(0.2, 0.8),
        opacity=0.7,
        interpolation="nearest",
        visibility=(True, False, True),
        initial_slices=(3, 5, 7),
    )

    # Verify basic properties
    assert isinstance(actor, Group)
    assert len(actor.children) == 3  # Should have 3 slice actors

    # Verify affine transform was applied correctly
    for child in actor.children:
        assert hasattr(child, "local")
        assert isinstance(child.local, AffineTransform)
        assert np.allclose(child.local.matrix, affine)

    # Verify visibility settings (only x and z should be visible)
    assert actor.children[0].visible is True  # x slice
    assert actor.children[1].visible is False  # y slice
    assert actor.children[2].visible is True  # z slice

    # Verify opacity
    for child in actor.children:
        assert child.material.opacity == pytest.approx(0.7)

    # Verify interpolation
    for child in actor.children:
        assert child.material.interpolation == "nearest"

    # Verify affine
    for child in actor.children:
        assert isinstance(child.local, AffineTransform)
        assert np.allclose(child.local.matrix, affine)


def test_peaks_slicer_basic_functionality():
    """Test basic functionality with minimal required inputs"""
    peak_dirs = np.random.rand(5, 5, 5, 3)  # 3D vector field
    result = peaks_slicer(peak_dirs)

    assert result is not None
    assert isinstance(result, Group)
    assert result.name == "VectorFieldSlicer"
    assert hasattr(result, "get_bounding_box")
    assert np.all(result.get_bounding_box() == np.array([[0, 0, 0], [4, 4, 4]]))
    vf = result.children[0]
    assert np.all(vf.cross_section == np.array([2, 2, 2]))
    assert vf.material.opacity == 1.0  # Default value


def test_peaks_slicer_with_affine_transform():
    """Test with affine transformation"""
    peak_dirs = np.random.rand(3, 3, 3, 3)
    affine = np.eye(4)
    affine[:3, 3] = [10, 20, 30]  # Translation

    result = peaks_slicer(peak_dirs, affine=affine)
    assert result is not None
    assert isinstance(result, Group)
    # Verify bounding box was transformed
    assert hasattr(result, "get_bounding_box")
    assert hasattr(result, "bounds")
    assert np.allclose(result.bounds[0], np.array([10, 20, 30]))
    assert np.allclose(result.bounds[1], np.array([12, 22, 32]))
    assert np.allclose(get_slices(result), np.array([11, 21, 31]))


def test_peaks_slicer_invalid_peak_dirs_shape():
    """Test with invalid peak directions shape"""
    with pytest.raises(ValueError):
        peaks_slicer(np.random.rand(3, 3))  # Not enough dimensions

    with pytest.raises(ValueError):
        peaks_slicer(np.random.rand(3, 3, 3, 3, 3, 3))  # Too many dimensions


def test_peaks_slicer_peak_values():
    """Test different peak_values configurations"""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    # Test single float value
    result1 = peaks_slicer(peak_dirs, peak_values=0.5)

    # Test array of values
    result2 = peaks_slicer(peak_dirs, peak_values=np.random.rand(2, 2, 2))

    assert result1 is not None
    assert result2 is not None


def test_peaks_slicer_actor_types():
    """Test different actor_type options"""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    result_thin = peaks_slicer(peak_dirs, actor_type="thin_line")
    result_thick = peaks_slicer(peak_dirs, actor_type="line", thickness=2.0)

    assert result_thin is not None
    assert result_thick is not None


def test_peaks_slicer_visual_properties():
    """Test visual properties (colors, opacity)"""
    peak_dirs = np.random.rand(2, 2, 2, 3)
    colors = np.random.rand(8, 3)

    result = peaks_slicer(
        peak_dirs, colors=colors, opacity=0.5, visibility=(True, False, True)
    )

    assert result is not None
    assert isinstance(result, Group)
    assert result.children[0].material.opacity == 0.5


def test_peaks_slicer_cross_section():
    """Test cross_section parameter"""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    result_default = peaks_slicer(peak_dirs)

    assert result_default is not None

    with pytest.raises((ValueError, TypeError)):
        peaks_slicer(peak_dirs, cross_section=0.0)


@pytest.mark.parametrize("affine", [None, np.diag([1, 1, 1, 1])])
def test_contour_from_roi(affine):
    """Test contour_from_roi with optional affine transformation."""
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 1

    contours = contour_from_roi(data, affine=affine)
    assert isinstance(contours, Group)
    assert len(contours.children) > 0
    actor = contours.children[0]
    assert np.allclose(actor.material.color[:3], [1, 0, 0])
    assert actor.material.opacity == pytest.approx(0.5)


@pytest.mark.parametrize(
    "colors,opacities",
    [
        (None, None),
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([0.8, 0.2])),
        (np.array([[0, 0, 1, 0.3]]), None),
    ],
)
def test_contour_from_label(colors, opacities):
    """Test contour_from_label with various color and opacity configurations."""
    data = np.zeros((5, 5, 5), dtype=int)
    data[1, 1, 1] = 1
    if colors is None or colors.shape[0] == 2:
        data[3, 3, 3] = 2

    contours = contour_from_label(data, colors=colors, opacities=opacities)
    assert isinstance(contours, Group)
    assert len(contours.children) == (colors.shape[0] if colors is not None else 2)

    if colors is not None and colors.shape[1] == 4:
        actor = contours.children[0].children[0]
        assert np.allclose(actor.material.color[:3], colors[0, :3])
        assert actor.material.opacity == pytest.approx(colors[0, 3])
    elif colors is not None:
        for i in range(len(contours.children)):
            actor = contours.children[i].children[0]
            assert np.allclose(actor.material.color[:3], colors[i])
            if opacities is not None:
                assert actor.material.opacity == pytest.approx(opacities[i])


@pytest.mark.parametrize(
    "colors,opacities,error_msg",
    [
        (np.array([[1, 0]]), None, "Incorrect color array shape"),
        (np.array([[1, 0, 0]]), np.array([0.5, 0.5]), "Incorrect opacity array shape"),
    ],
)
def test_contour_from_label_errors(colors, opacities, error_msg):
    """Test contour_from_label error conditions."""
    data = np.zeros((5, 5, 5), dtype=int)
    data[1, 1, 1] = 1

    with pytest.raises(ValueError, match=error_msg):
        contour_from_label(data, colors=colors, opacities=opacities)
