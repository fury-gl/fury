"""Core module testing."""

import numpy.testing as npt
import warnings

from fury import window, ui


def test_ui_rectangle_2d():
    window_size = (700, 700)
    show_manager = window.ShowManager(size=window_size)

    rect = ui.Rectangle2D(size=(100, 50))
    rect.position = (50, 80)
    npt.assert_equal(rect.position, (50, 80))

    rect.color = (1, 0.5, 0)
    npt.assert_equal(rect.color, (1, 0.5, 0))

    rect.opacity = 0.5
    npt.assert_equal(rect.opacity, 0.5)

    # Check the rectangle is drawn at right place.
    show_manager.scene.add(rect)
    # Uncomment this to start the visualisation
    # show_manager.start()

    colors = [rect.color]
    arr = window.snapshot(show_manager.scene, size=window_size, offscreen=True)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [False])

    # Test visibility off.
    rect.set_visibility(False)
    arr = window.snapshot(show_manager.scene, size=window_size, offscreen=True)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 0)


def test_ui_disk_2d():
    window_size = (700, 700)
    show_manager = window.ShowManager(size=window_size)

    disk = ui.Disk2D(outer_radius=20, inner_radius=5)
    disk.position = (50, 80)
    npt.assert_equal(disk.position, (50, 80))

    disk.color = (1, 0.5, 0)
    npt.assert_equal(disk.color, (1, 0.5, 0))

    disk.opacity = 0.5
    npt.assert_equal(disk.opacity, 0.5)

    # Check the rectangle is drawn at right place.
    show_manager.scene.add(disk)
    # Uncomment this to start the visualisation
    # show_manager.start()

    colors = [disk.color]
    arr = window.snapshot(show_manager.scene, size=window_size, offscreen=True)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 1)
    # Should be False because of the offscreen
    npt.assert_equal(report.colors_found, [False])

    # Test visibility off.
    disk.set_visibility(False)
    arr = window.snapshot(show_manager.scene, size=window_size, offscreen=True)
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 0)


def test_text_block_2d():
    text_block = ui.TextBlock2D()

    def _check_property(obj, attr, values):
        for value in values:
            setattr(obj, attr, value)
            npt.assert_equal(getattr(obj, attr), value)

    _check_property(text_block, "bold", [True, False])
    _check_property(text_block, "italic", [True, False])
    _check_property(text_block, "shadow", [True, False])
    _check_property(text_block, "font_size", range(100))
    _check_property(text_block, "message", ["", "Hello World", "Line\nBreak"])
    _check_property(text_block, "justification", ["left", "center", "right"])
    _check_property(text_block, "position", [(350, 350), (0.5, 0.5)])
    _check_property(text_block, "color", [(0., 0.5, 1.)])
    _check_property(text_block, "background_color", [(0., 0.5, 1.), None])
    _check_property(text_block, "vertical_justification",
                    ["top", "middle", "bottom"])
    _check_property(text_block, "font_family", ["Arial", "Courier"])

    with npt.assert_raises(ValueError):
        text_block.font_family = "Verdana"

    with npt.assert_raises(ValueError):
        text_block.justification = "bottom"

    with npt.assert_raises(ValueError):
        text_block.vertical_justification = "left"


def test_text_block_2d_justification():
    window_size = (700, 700)
    show_manager = window.ShowManager(size=window_size)

    # To help visualize the text positions.
    grid_size = (500, 500)
    bottom, middle, top = 50, 300, 550
    left, center, right = 50, 300, 550
    line_color = (1, 0, 0)

    grid_top = (center, top), (grid_size[0], 1)
    grid_bottom = (center, bottom), (grid_size[0], 1)
    grid_left = (left, middle), (1, grid_size[1])
    grid_right = (right, middle), (1, grid_size[1])
    grid_middle = (center, middle), (grid_size[0], 1)
    grid_center = (center, middle), (1, grid_size[1])
    grid_specs = [grid_top, grid_bottom, grid_left, grid_right,
                  grid_middle, grid_center]
    for spec in grid_specs:
        line = ui.Rectangle2D(size=spec[1], color=line_color)
        line.center = spec[0]
        show_manager.scene.add(line)

    font_size = 60
    bg_color = (1, 1, 1)
    texts = []
    texts += [ui.TextBlock2D("HH", position=(left, top),
                             font_size=font_size,
                             color=(1, 0, 0), bg_color=bg_color,
                             justification="left",
                             vertical_justification="top")]
    texts += [ui.TextBlock2D("HH", position=(center, top),
                             font_size=font_size,
                             color=(0, 1, 0), bg_color=bg_color,
                             justification="center",
                             vertical_justification="top")]
    texts += [ui.TextBlock2D("HH", position=(right, top),
                             font_size=font_size,
                             color=(0, 0, 1), bg_color=bg_color,
                             justification="right",
                             vertical_justification="top")]

    texts += [ui.TextBlock2D("HH", position=(left, middle),
                             font_size=font_size,
                             color=(1, 1, 0), bg_color=bg_color,
                             justification="left",
                             vertical_justification="middle")]
    texts += [ui.TextBlock2D("HH", position=(center, middle),
                             font_size=font_size,
                             color=(0, 1, 1), bg_color=bg_color,
                             justification="center",
                             vertical_justification="middle")]
    texts += [ui.TextBlock2D("HH", position=(right, middle),
                             font_size=font_size,
                             color=(1, 0, 1), bg_color=bg_color,
                             justification="right",
                             vertical_justification="middle")]

    texts += [ui.TextBlock2D("HH", position=(left, bottom),
                             font_size=font_size,
                             color=(0.5, 0, 1), bg_color=bg_color,
                             justification="left",
                             vertical_justification="bottom")]
    texts += [ui.TextBlock2D("HH", position=(center, bottom),
                             font_size=font_size,
                             color=(1, 0.5, 0), bg_color=bg_color,
                             justification="center",
                             vertical_justification="bottom")]
    texts += [ui.TextBlock2D("HH", position=(right, bottom),
                             font_size=font_size,
                             color=(0, 1, 0.5), bg_color=bg_color,
                             justification="right",
                             vertical_justification="bottom")]

    show_manager.scene.add(*texts)

    # Uncomment this to start the visualisation
    # show_manager.start()

    window.snapshot(show_manager.scene, size=window_size, offscreen=True)


def test_text_block_2d_size():
    text_block_1 = ui.TextBlock2D(position=(50, 50), size=(100, 100))

    npt.assert_equal(text_block_1.actor.GetTextScaleMode(), 1)
    npt.assert_equal(text_block_1.size, (100, 100))

    text_block_1.font_size = 50

    npt.assert_equal(text_block_1.actor.GetTextScaleMode(), 0)
    npt.assert_equal(text_block_1.font_size, 50)

    text_block_2 = ui.TextBlock2D(position=(50, 50), font_size=50)

    npt.assert_equal(text_block_2.actor.GetTextScaleMode(), 0)
    npt.assert_equal(text_block_2.font_size, 50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        text_block_2.size
        npt.assert_equal(len(w), 1)
        npt.assert_(issubclass(w[-1].category, RuntimeWarning))

    text_block_2.resize((100, 100))

    npt.assert_equal(text_block_2.actor.GetTextScaleMode(), 1)
    npt.assert_equal(text_block_2.size, (100, 100))

    text_block_2.position = (100, 100)
    npt.assert_equal(text_block_2.position, (100, 100))

    window_size = (700, 700)
    show_manager = window.ShowManager(size=window_size)

    text_block_3 = ui.TextBlock2D(text="FURY\nFURY\nFURY\nHello",
                                  position=(150, 100), bg_color=(1, 0, 0),
                                  color=(0, 1, 0), size=(100, 100))

    show_manager.scene.add(text_block_3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        text_block_3.font_size = 100
        npt.assert_equal(len(w), 1)
        npt.assert_(issubclass(w[-1].category, RuntimeWarning))
        npt.assert_equal(text_block_3.size, (100, 100))

        text_block_3.font_size = 12
        npt.assert_equal(len(w), 1)
        npt.assert_(issubclass(w[-1].category, RuntimeWarning))
        npt.assert_equal(text_block_3.font_size, 12)
