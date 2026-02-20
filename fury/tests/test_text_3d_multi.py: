import numpy as np
import pytest
from fury import actor
from fury.lib import Assembly


def test_text_3d_single():
    """Single string should return Text3D actor (backward compatible)."""
    t = actor.text_3d("Hello", position=(0, 0, 0))
    assert t is not None


def test_text_3d_multiple():
    """List of strings should return Assembly."""
    texts = ["Hello", "World", "FURY"]
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    t = actor.text_3d(texts, position=positions)
    assert isinstance(t, Assembly)


def test_text_3d_multiple_colors():
    """Multiple texts with individual colors."""
    texts = ["Red", "Green", "Blue"]
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    t = actor.text_3d(texts, position=positions, color=colors)
    assert isinstance(t, Assembly)


def test_text_3d_multiple_font_sizes():
    """Multiple texts with individual font sizes."""
    texts = ["Small", "Big"]
    positions = [(0, 0, 0), (1, 0, 0)]
    t = actor.text_3d(texts, position=positions, font_size=[10, 24])
    assert isinstance(t, Assembly)
