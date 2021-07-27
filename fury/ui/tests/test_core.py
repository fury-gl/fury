"""Core module testing."""
from os.path import join as pjoin
import numpy as np
import numpy.testing as npt
import warnings

from fury.data import DATA_DIR, read_viz_icons, fetch_viz_icons
from fury import window, ui
from fury.testing import EventCounter


def test_ui_button_panel(recording=False):
    filename = "test_ui_button_panel"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    # Rectangle
    rectangle_test = ui.Rectangle2D(size=(10, 10))
    another_rectangle_test = ui.Rectangle2D(size=(1, 1))

    # Button
    fetch_viz_icons()

    icon_files = []
    icon_files.append(('stop', read_viz_icons(fname='stop2.png')))
    icon_files.append(('play', read_viz_icons(fname='play3.png')))

    button_test = ui.Button2D(icon_fnames=icon_files)
    button_test.center = (20, 20)

    def make_invisible(i_ren, _obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.set_visibility(False)
        i_ren.force_render()
        i_ren.event.abort()

    def modify_button_callback(i_ren, _obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.next_icon()
        i_ren.force_render()

    button_test.on_right_mouse_button_pressed = make_invisible
    button_test.on_left_mouse_button_pressed = modify_button_callback

    button_test.scale((2, 2))
    button_color = button_test.color
    button_test.color = button_color

    # TextBlock
    text_block_test = ui.TextBlock2D()
    text_block_test.message = 'TextBlock'
    text_block_test.color = (0, 0, 0)

    # Panel
    panel = ui.Panel2D(size=(300, 150),
                       position=(290, 15),
                       color=(1, 1, 1), align="right",
                       has_border=True)
    
    non_bordered_panel = ui.Panel2D(size=(100, 100),
                                    has_border=False)

    npt.assert_equal(hasattr(non_bordered_panel, 'borders'), False)

    panel.add_element(rectangle_test, (290, 135))
    panel.add_element(button_test, (0.1, 0.1))
    panel.add_element(text_block_test, (0.7, 0.7))
    npt.assert_raises(ValueError, panel.add_element, another_rectangle_test,
                      (10., 0.5))
    npt.assert_raises(ValueError, panel.add_element, another_rectangle_test,
                      (-0.5, 0.5))

    npt.assert_equal(panel.border_width, [0.0, ]*4)
    npt.assert_equal(panel.border_color, [np.asarray([1, 1, 1]), ]*4)

    panel.border_width = ['bottom', 10.0]
    npt.assert_equal(panel.border_width[3], 10.0)
    npt.assert_equal(panel.borders['bottom'].height, 10.0)

    panel.border_width = ['right', 10.0]
    npt.assert_equal(panel.border_width[1], 10.0)
    npt.assert_equal(panel.borders['right'].width, 10.0)

    with npt.assert_raises(ValueError):
        panel.border_width = ['invalid_label', 10.0]

    panel.border_color = ['bottom', (0.4, 0.5, 0.6)]
    npt.assert_equal(panel.border_color[3], (0.4, 0.5, 0.6))

    with npt.assert_raises(ValueError):
        panel.border_color = ['invalid_label', (0.4, 0.5, 0.6)]

    new_size = (400, 400)
    panel.resize(new_size)
    npt.assert_equal(panel.borders['bottom'].width, 400.0)
    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(button_test)
    event_counter.monitor(panel.background)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="FURY Button")

    show_manager.scene.add(panel)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


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
