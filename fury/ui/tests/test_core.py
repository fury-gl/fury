"""Core module testing."""

from unittest.mock import patch

import numpy.testing as npt
import pytest

from fury import ui
from fury.lib import Mesh
from fury.ui.helpers import Anchor


@pytest.fixture
def mock_ui_context_v1():
    """Mock UIContext for V1 UI (y_anchor BOTTOM)."""
    with (
        patch("fury.ui.UIContext.get_is_v2_ui", return_value=False),
        patch("fury.ui.UIContext.get_canvas_size", return_value=(800, 600)),
    ):  # Consistent canvas size for positioning
        yield ui.UIContext


@pytest.fixture
def mock_ui_context_v2():
    """Mock UIContext for V2 UI (y_anchor TOP)."""
    with (
        patch("fury.ui.UIContext.get_is_v2_ui", return_value=True),
        patch("fury.ui.UIContext.get_canvas_size", return_value=(800, 600)),
    ):  # Consistent canvas size for positioning
        yield ui.UIContext


def test_rectangle2d_initialization_default(mock_ui_context_v1):
    """
    Test Rectangle2D initialization with default parameters.
    Checks default size, color, opacity, and position.
    """
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.size, (0, 0))  # Default size
    npt.assert_array_equal(
        rect.color, [1, 1, 1, 1]
    )  # Default color (white, with alpha)
    npt.assert_equal(rect.opacity, 1.0)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors
    # Default position (0,0) with LEFT, BOTTOM anchor for V1 UI
    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.BOTTOM), [0, 0])


def test_rectangle2d_initialization_custom(mock_ui_context_v1):
    """
    Test Rectangle2D initialization with custom parameters.
    Checks custom size, position, color, and opacity.
    """
    custom_size = (200, 100)
    custom_position = (50, 80)
    custom_color = (0.5, 0.2, 0.8)
    custom_opacity = 0.7

    rect = ui.Rectangle2D(
        size=custom_size,
        position=custom_position,
        color=custom_color,
        opacity=custom_opacity,
    )

    npt.assert_equal(rect.size, custom_size)
    npt.assert_array_equal(
        rect.get_position(Anchor.LEFT, Anchor.BOTTOM), custom_position
    )
    npt.assert_array_almost_equal(rect.color[:3], custom_color)  # Check RGB part
    npt.assert_almost_equal(rect.opacity, custom_opacity)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors


def test_rectangle2d_width_property(mock_ui_context_v1):
    """Test width getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.width, 100)
    rect.width = 120
    npt.assert_equal(rect.width, 120)
    npt.assert_equal(rect.height, 50)  # Height should remain unchanged
    npt.assert_equal(rect.size, (120, 50))


def test_rectangle2d_height_property(mock_ui_context_v1):
    """Test height getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.height, 50)
    rect.height = 70
    npt.assert_equal(rect.height, 70)
    npt.assert_equal(rect.width, 100)  # Width should remain unchanged
    npt.assert_equal(rect.size, (100, 70))


def test_rectangle2d_color_property(mock_ui_context_v1):
    """Test color getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_array_equal(rect.color, [1, 1, 1, 1])  # Default color
    new_color = (0.1, 0.2, 0.3)
    rect.color = new_color
    npt.assert_array_almost_equal(rect.color[:3], new_color)


def test_rectangle2d_opacity_property(mock_ui_context_v1):
    """Test opacity getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.opacity, 1.0)
    new_opacity = 0.5
    rect.opacity = new_opacity
    npt.assert_equal(rect.opacity, new_opacity)


def test_rectangle2d_resize(mock_ui_context_v1):
    """Test resize method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    new_size = (250, 150)
    rect.resize(new_size)
    npt.assert_equal(rect.size, new_size)


def test_rectangle2d_set_visibility(mock_ui_context_v1):
    """Test set_visibility method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(10, 10))
    rect.set_visibility(False)
    assert rect.actor.visible is False
    rect.set_visibility(True)
    assert rect.actor.visible is True


def test_disk2d_initialization_default(mock_ui_context_v1):
    """
    Test Disk2D initialization with default parameters.
    Checks default center, color, opacity, and required outer_radius.
    """
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_array_equal(
        disk_ui.get_position(Anchor.CENTER, Anchor.CENTER), [0, 0]
    )  # Default center
    npt.assert_array_equal(
        disk_ui.color, [1, 1, 1, 1]
    )  # Default color (white, with alpha)
    npt.assert_equal(disk_ui.opacity, 1.0)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (20, 20))  # Diameter is 2 * radius


def test_disk2d_initialization_custom(mock_ui_context_v1):
    """
    Test Disk2D initialization with custom parameters.
    Checks custom outer_radius, center, color, and opacity.
    """
    custom_radius = 25
    custom_center = (100, 150)
    custom_color = (0.9, 0.1, 0.4)
    custom_opacity = 0.6

    disk_ui = ui.Disk2D(
        outer_radius=custom_radius,
        center=custom_center,
        color=custom_color,
        opacity=custom_opacity,
    )

    npt.assert_equal(disk_ui.outer_radius, custom_radius)
    npt.assert_array_equal(
        disk_ui.get_position(Anchor.CENTER, Anchor.CENTER), custom_center
    )
    npt.assert_array_almost_equal(disk_ui.color[:3], custom_color)
    npt.assert_almost_equal(disk_ui.opacity, custom_opacity)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (50, 50))  # Diameter is 2 * radius


def test_disk2d_outer_radius_property(mock_ui_context_v1):
    """Test outer_radius getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_equal(disk_ui.size, (20, 20))  # Diameter

    disk_ui.outer_radius = 15
    npt.assert_equal(disk_ui.outer_radius, 15)
    npt.assert_equal(disk_ui.size, (30, 30))  # New diameter


def test_disk2d_color_property(mock_ui_context_v1):
    """Test color getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_array_almost_equal(disk_ui.color, [1, 1, 1, 1])
    new_color = (0.6, 0.7, 0.8)
    disk_ui.color = new_color
    npt.assert_array_almost_equal(disk_ui.color[:3], new_color)


def test_disk2d_opacity_property(mock_ui_context_v1):
    """Test opacity getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_almost_equal(disk_ui.opacity, 1.0)
    new_opacity = 0.3
    disk_ui.opacity = new_opacity
    npt.assert_almost_equal(disk_ui.opacity, new_opacity)


def test_disk2d_set_visibility(mock_ui_context_v1):
    """Test set_visibility method for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    disk_ui.set_visibility(False)
    assert disk_ui.actor.visible is False
    disk_ui.set_visibility(True)
    assert disk_ui.actor.visible is True
