"""Core module testing."""

from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.actor import Mesh
from fury.ui.helpers import Anchor


def test_rectangle2d_initialization_default():
    """
    Test Rectangle2D initialization with default parameters.
    Checks default size, color, opacity, and position.
    """
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.size, (100, 100))
    npt.assert_array_equal(rect.color, [1, 1, 1])
    npt.assert_equal(rect.opacity, 1.0)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors

    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.BOTTOM), [0, 100])


def test_rectangle2d_initialization_custom():
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
    npt.assert_array_equal(rect.get_position(Anchor.LEFT, Anchor.TOP), custom_position)
    npt.assert_array_almost_equal(rect.color, custom_color)
    npt.assert_almost_equal(rect.opacity, custom_opacity)
    assert isinstance(rect.actor, Mesh)
    assert rect.actor in rect.actors


def test_rectangle2d_width_property():
    """Test width getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.width, 100)
    rect.width = 120
    npt.assert_equal(rect.width, 120)
    npt.assert_equal(rect.height, 50)
    npt.assert_equal(rect.size, (120, 50))


def test_rectangle2d_height_property():
    """Test height getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    npt.assert_equal(rect.height, 50)
    rect.height = 70
    npt.assert_equal(rect.height, 70)
    npt.assert_equal(rect.width, 100)
    npt.assert_equal(rect.size, (100, 70))


def test_rectangle2d_color_property():
    """Test color getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_array_equal(rect.color, [1, 1, 1])
    new_color = (0.1, 0.2, 0.3)
    rect.color = new_color
    npt.assert_array_almost_equal(rect.color, new_color)


def test_rectangle2d_opacity_property():
    """Test opacity getter and setter for Rectangle2D."""
    rect = ui.Rectangle2D()
    npt.assert_equal(rect.opacity, 1.0)
    new_opacity = 0.5
    rect.opacity = new_opacity
    npt.assert_equal(rect.opacity, new_opacity)


def test_rectangle2d_resize():
    """Test resize method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(100, 50))
    new_size = (250, 150)
    rect.resize(new_size)
    npt.assert_equal(rect.size, new_size)


def test_rectangle2d_set_visibility():
    """Test set_visibility method for Rectangle2D."""
    rect = ui.Rectangle2D(size=(10, 10))
    rect.set_visibility(False)
    assert rect.actor.visible is False
    rect.set_visibility(True)
    assert rect.actor.visible is True


def test_disk2d_initialization_default():
    """
    Test Disk2D initialization with default parameters.
    Checks default center, color, opacity, and required outer_radius.
    """
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_array_equal(disk_ui.get_position(Anchor.CENTER, Anchor.CENTER), [0, 0])
    npt.assert_array_equal(disk_ui.color, [1, 1, 1])
    npt.assert_equal(disk_ui.opacity, 1.0)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (20, 20))


def test_disk2d_initialization_custom():
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
    npt.assert_array_almost_equal(disk_ui.color, custom_color)
    npt.assert_almost_equal(disk_ui.opacity, custom_opacity)
    assert isinstance(disk_ui.actor, Mesh)
    assert disk_ui.actor in disk_ui.actors
    npt.assert_equal(disk_ui.size, (50, 50))


def test_disk2d_outer_radius_property():
    """Test outer_radius getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_equal(disk_ui.outer_radius, 10)
    npt.assert_equal(disk_ui.size, (20, 20))

    disk_ui.outer_radius = 15
    npt.assert_equal(disk_ui.outer_radius, 15)
    npt.assert_equal(disk_ui.size, (30, 30))


def test_disk2d_inner_radius_property():
    """Test inner_radius getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=20, inner_radius=10)
    npt.assert_equal(disk_ui.inner_radius, 10)

    disk_ui.inner_radius = 15
    npt.assert_equal(disk_ui.inner_radius, 15)

    with npt.assert_raises(ValueError):
        disk_ui.inner_radius = 25


def test_disk2d_color_property():
    """Test color getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_array_almost_equal(disk_ui.color, [1, 1, 1])
    new_color = (0.6, 0.7, 0.8)
    disk_ui.color = new_color
    npt.assert_array_almost_equal(disk_ui.color, new_color)


def test_disk2d_opacity_property():
    """Test opacity getter and setter for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    npt.assert_almost_equal(disk_ui.opacity, 1.0)
    new_opacity = 0.3
    disk_ui.opacity = new_opacity
    npt.assert_almost_equal(disk_ui.opacity, new_opacity)


def test_disk2d_set_visibility():
    """Test set_visibility method for Disk2D."""
    disk_ui = ui.Disk2D(outer_radius=10)
    disk_ui.set_visibility(False)
    assert disk_ui.actor.visible is False
    disk_ui.set_visibility(True)
    assert disk_ui.actor.visible is True


def test_rectangle2d_visual_snapshot():
    """Visual test for Rectangle2D."""
    rect_size = (50, 50)
    rect_pos_ui = (75, 75)
    rect_color = (1.0, 0.0, 0.0)

    rect = ui.Rectangle2D(
        size=rect_size,
        position=rect_pos_ui,
        color=rect_color,
        opacity=1.0,
    )

    scene = window.Scene()
    scene.add(rect)

    fname = "rect_test_visible.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    npt.assert_almost_equal(mean_g, 0, decimal=0)
    npt.assert_almost_equal(mean_b, 0, decimal=0)
    assert 0 < mean_r <= 255

    scene = window.Scene()
    scene.add(rect)

    rect.set_visibility(False)
    fname_hidden = "rect_test_hidden.png"
    window.snapshot(scene=scene, fname=fname_hidden)

    img_hidden = Image.open(fname_hidden)
    img_array_hidden = np.array(img_hidden)

    mean_r_hidden, mean_g_hidden, mean_b_hidden, _mean_a_hidden = np.mean(
        img_array_hidden.reshape(-1, img_array_hidden.shape[2]), axis=0
    )
    npt.assert_almost_equal(mean_r_hidden, 0, decimal=0)
    npt.assert_almost_equal(mean_g_hidden, 0, decimal=0)
    npt.assert_almost_equal(mean_b_hidden, 0, decimal=0)


def test_disk2d_visual_snapshot():
    """Visual test for Disk2D."""
    disk_radius = 25
    disk_center_ui = (100, 100)
    disk_color = (0.0, 1.0, 0.0)

    disk = ui.Disk2D(
        outer_radius=disk_radius,
        center=disk_center_ui,
        color=disk_color,
    )

    scene = window.Scene()
    scene.add(disk)

    fname = "disk_test_visible.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_g > mean_r and mean_g > mean_b

    npt.assert_almost_equal(mean_r, 0, decimal=0)
    npt.assert_almost_equal(mean_b, 0, decimal=0)
    assert 0 < mean_g <= 255

    scene = window.Scene()
    scene.add(disk)

    disk.set_visibility(False)
    fname_hidden = "disk_test_hidden.png"
    window.snapshot(scene=scene, fname=fname_hidden)

    img_hidden = Image.open(fname_hidden)
    img_array_hidden = np.array(img_hidden)

    mean_r_hidden, mean_g_hidden, mean_b_hidden, _mean_a_hidden = np.mean(
        img_array_hidden.reshape(-1, img_array_hidden.shape[2]), axis=0
    )
    npt.assert_almost_equal(mean_r_hidden, 0, decimal=0)
    npt.assert_almost_equal(mean_g_hidden, 0, decimal=0)
    npt.assert_almost_equal(mean_b_hidden, 0, decimal=0)


def test_textblock2d_initialization_default():
    """
    Test TextBlock2D initialization with minimal valid parameters.
    Checks default font, color, alignment, and flags.
    """
    tb = ui.TextBlock2D(size=(200, 100))

    npt.assert_equal(tb.message, "Text Block")
    npt.assert_equal(tb.font_size, 18)
    npt.assert_equal(tb.font_family[0], "Arial")
    npt.assert_array_equal(tb.color, [1, 1, 1])

    assert tb.bold is False
    assert tb.italic is False
    assert tb.have_bg is False
    assert tb.background_color is None

    assert tb.justification == "left"
    assert tb.vertical_justification == "bottom"


def test_textblock2d_initialization_custom():
    """Test TextBlock2D initialization with full custom parameters."""
    custom_color = (1.0, 1.0, 0.0)
    custom_bg = (0.0, 0.0, 1.0)

    tb = ui.TextBlock2D(
        text="Custom",
        font_size=24,
        font_family="Times New Roman",
        justification="center",
        vertical_justification="top",
        bold=True,
        italic=False,
        size=(300, 150),
        color=custom_color,
        bg_color=custom_bg,
        position=(10, 10),
    )

    npt.assert_equal(tb.message, "Custom")
    npt.assert_equal(tb.font_size, 24)
    npt.assert_equal(tb.font_family[0], "Times New Roman")
    npt.assert_array_almost_equal(tb.color, custom_color)

    assert tb.have_bg is True
    npt.assert_array_almost_equal(tb.background_color, custom_bg)
    assert isinstance(tb.background, ui.Rectangle2D)

    assert "**Custom**" in tb.get_formatted_text("Custom")


def test_textblock2d_message_property():
    """Test message getter and setter."""
    tb = ui.TextBlock2D(size=(100, 50))
    npt.assert_equal(tb.message, "Text Block")

    new_msg = "New Message"
    tb.message = new_msg
    npt.assert_equal(tb.message, new_msg)


def test_textblock2d_font_properties():
    """Test font_size and font_family properties."""
    tb = ui.TextBlock2D(size=(100, 50))

    tb.font_size = 30
    npt.assert_equal(tb.font_size, 30)

    tb.font_family = "Courier"
    npt.assert_equal(tb.font_family[0], "Courier")


def test_textblock2d_style_properties():
    """Test bold and italic properties."""
    tb = ui.TextBlock2D(text="Style", size=(100, 50))

    assert tb.get_formatted_text("Style") == "Style"

    tb.bold = True
    assert tb.bold is True
    assert tb.get_formatted_text("Style") == "**Style**"

    tb.bold = False
    tb.italic = True
    assert tb.italic is True
    assert tb.get_formatted_text("Style") == "*Style*"


def test_textblock2d_alignment_valid():
    """Test valid inputs for justification properties."""
    tb = ui.TextBlock2D(size=(100, 100))

    valid_h = ["left", "center", "right"]
    valid_v = ["top", "middle", "bottom"]

    for h in valid_h:
        tb.justification = h
        npt.assert_equal(tb.justification, h)

    for v in valid_v:
        tb.vertical_justification = v
        npt.assert_equal(tb.vertical_justification, v)


def test_textblock2d_alignment_invalid():
    """Test invalid inputs for justification properties."""
    tb = ui.TextBlock2D(size=(100, 100))

    with npt.assert_raises(ValueError):
        tb.justification = "diagonal"

    with npt.assert_raises(ValueError):
        tb.vertical_justification = "sideways"


def test_textblock2d_color_property():
    """Test text color getter and setter."""
    tb = ui.TextBlock2D(size=(100, 50))
    npt.assert_array_equal(tb.color, [1, 1, 1])

    tb.color = (0.5, 0.5, 0.5)
    npt.assert_array_almost_equal(tb.color, [0.5, 0.5, 0.5])


def test_textblock2d_background_color_property():
    """Test background color logic."""
    tb = ui.TextBlock2D(size=(100, 50))

    assert tb.background_color is None
    assert tb.have_bg is False

    tb.background_color = (1, 0, 0)
    assert tb.have_bg is True
    npt.assert_array_equal(tb.background_color, (1, 0, 0))
    assert tb.background.actor.visible is True

    tb.background_color = None
    assert tb.have_bg is False
    assert tb.background_color is None
    assert tb.background.actor.visible is False


def test_textblock2d_dynamic_bbox_property():
    """Test dynamic_bbox property toggling."""
    tb = ui.TextBlock2D(size=(100, 50))
    assert tb.dynamic_bbox is False

    tb.dynamic_bbox = True
    assert tb.dynamic_bbox is True

    size = tb._get_size()
    assert len(size) == 2


def test_textblock2d_resize():
    """Test resize method."""
    tb = ui.TextBlock2D(size=(100, 50))
    new_size = (200, 150)

    tb.resize(new_size)

    npt.assert_equal(tb.background.size, new_size)

    npt.assert_equal(tb.boundingbox[2] - tb.boundingbox[0], new_size[0])
    npt.assert_equal(tb.boundingbox[3] - tb.boundingbox[1], new_size[1])


def test_textblock2d_visual_snapshot():
    """
    Visual test for TextBlock2D.
    Checks if background renders correctly when enabled.
    """

    tb = ui.TextBlock2D(
        text="Visual Test",
        size=(200, 100),
        position=(50, 50),
        bg_color=(1.0, 0.0, 0.0),
        color=(1.0, 1.0, 1.0),
    )

    scene = window.Scene()
    scene.add(tb)

    fname = "textblock_visible.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_color = np.mean(img_array.reshape(-1, img_array.shape[2]), axis=0)

    assert mean_color[0] > 0.5

    assert mean_color[0] > mean_color[2]

    scene = window.Scene()
    scene.add(tb)

    if hasattr(tb, "set_visibility"):
        tb.set_visibility(False)
        fname_hidden = "textblock_hidden.png"
        window.snapshot(scene=scene, fname=fname_hidden)

        img_hidden = Image.open(fname_hidden)
        arr_hidden = np.array(img_hidden)

        mean_hidden = np.mean(arr_hidden[:, :, :3])
        npt.assert_almost_equal(mean_hidden, 0, decimal=0)
