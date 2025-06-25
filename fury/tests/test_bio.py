import numpy as np
import pytest

from fury.actor import volume_slicer
from fury.lib import AffineTransform, Group


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
