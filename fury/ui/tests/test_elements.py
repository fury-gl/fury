"""Test for components module."""
from os.path import join as pjoin
import os
import itertools

import numpy as np

import numpy.testing as npt
import pytest

from fury import window, actor, ui
from fury.data import DATA_DIR, read_viz_icons, fetch_viz_icons
from fury.decorators import skip_win, skip_osx
from fury.primitive import prim_sphere
from fury.testing import assert_arrays_equal, assert_greater, EventCounter


def test_ui_textbox(recording=False):
    filename = "test_ui_textbox"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    # TextBox
    textbox_test = ui.TextBox2D(height=3, width=10, text="Text")

    another_textbox_test = ui.TextBox2D(height=3, width=10, text="Enter Text")
    another_textbox_test.set_message("Enter Text")

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(textbox_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="FURY TextBox")

    show_manager.scene.add(textbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_line_slider_2d_horizontal_bottom(recording=False):
    filename = "test_ui_line_slider_2d_horizontal_bottom"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    line_slider_2d_test = ui.LineSlider2D(initial_value=-2,
                                          min_value=-5, max_value=5,
                                          orientation="horizontal",
                                          text_alignment='bottom')
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="FURY Horizontal Line Slider")

    show_manager.scene.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_line_slider_2d_horizontal_top(recording=False):
    filename = "test_ui_line_slider_2d_horizontal_top"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    line_slider_2d_test = ui.LineSlider2D(initial_value=-2,
                                          min_value=-5, max_value=5,
                                          orientation="horizontal",
                                          text_alignment='top')
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="FURY Horizontal Line Slider")

    show_manager.scene.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_line_slider_2d_vertical_left(recording=False):
    filename = "test_ui_line_slider_2d_vertical_left"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    line_slider_2d_test = ui.LineSlider2D(initial_value=-2,
                                          min_value=-5, max_value=5,
                                          orientation="vertical",
                                          text_alignment='left')
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="FURY Vertical Line Slider")

    show_manager.scene.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_line_slider_2d_vertical_right(recording=False):
    filename = "test_ui_line_slider_2d_vertical_right"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    line_slider_2d_test = ui.LineSlider2D(initial_value=-2,
                                          min_value=-5, max_value=5,
                                          orientation="vertical",
                                          text_alignment='right')
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="FURY Vertical Line Slider")

    show_manager.scene.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_line_double_slider_2d(interactive=False):
    line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
        center=(300, 300), shape="disk", outer_radius=15, min_value=-10,
        max_value=10, initial_values=(-10, 10))
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.handles[0].size, (30, 30))
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.left_disk_value, -10)
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.right_disk_value, 10)

    line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
        center=(300, 300), shape="disk", outer_radius=15, min_value=-10,
        max_value=10, initial_values=(-10, 10))
    npt.assert_equal(
        line_double_slider_2d_vertical_test.handles[0].size, (30, 30))
    npt.assert_equal(
        line_double_slider_2d_vertical_test.bottom_disk_value, -10)
    npt.assert_equal(
        line_double_slider_2d_vertical_test.top_disk_value, 10)

    if interactive:
        show_manager = window.ShowManager(size=(600, 600),
                                          title="FURY Line Double Slider")
        show_manager.scene.add(line_double_slider_2d_horizontal_test)
        show_manager.scene.add(line_double_slider_2d_vertical_test)
        show_manager.start()

    line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
        center=(300, 300), shape="square", handle_side=5,
        orientation="horizontal", initial_values=(50, 40))
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.handles[0].size, (5, 5))
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.left_disk_value, 39)
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.right_disk_value, 40)
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.left_disk_ratio, 0.39)
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.right_disk_ratio, 0.4)

    line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
        center=(300, 300), shape="square", handle_side=5,
        orientation="vertical", initial_values=(50, 40))
    npt.assert_equal(
        line_double_slider_2d_vertical_test.handles[0].size, (5, 5))
    npt.assert_equal(
        line_double_slider_2d_vertical_test.bottom_disk_value, 39)
    npt.assert_equal(
        line_double_slider_2d_vertical_test.top_disk_value, 40)
    npt.assert_equal(
        line_double_slider_2d_vertical_test.bottom_disk_ratio, 0.39)
    npt.assert_equal(
        line_double_slider_2d_vertical_test.top_disk_ratio, 0.4)

    with npt.assert_raises(ValueError):
        ui.LineDoubleSlider2D(orientation="Not_hor_not_vert")

    if interactive:
        show_manager = window.ShowManager(size=(600, 600),
                                          title="FURY Line Double Slider")
        show_manager.scene.add(line_double_slider_2d_horizontal_test)
        show_manager.scene.add(line_double_slider_2d_vertical_test)
        show_manager.start()


def test_ui_ring_slider_2d(recording=False):
    filename = "test_ui_ring_slider_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    ring_slider_2d_test = ui.RingSlider2D()
    ring_slider_2d_test.center = (300, 300)
    ring_slider_2d_test.value = 90

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(ring_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="FURY Ring Slider")

    show_manager.scene.add(ring_slider_2d_test)

    if recording:
        # Record the following events
        # 1. Left Click on the handle and hold it
        # 2. Move to the left the handle and make 1.5 tour
        # 3. Release the handle
        # 4. Left Click on the handle and hold it
        # 5. Move to the right the handle and make 1 tour
        # 6. Release the handle
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_range_slider(interactive=False):
    range_slider_test_horizontal = ui.RangeSlider(shape="square")
    range_slider_test_vertical = ui.RangeSlider(shape="square",
                                                orientation="vertical")

    if interactive:
        show_manager = window.ShowManager(size=(600, 600),
                                          title="FURY Line Double Slider")
        show_manager.scene.add(range_slider_test_horizontal)
        show_manager.scene.add(range_slider_test_vertical)
        show_manager.start()


def test_ui_option(interactive=False):
    option_test = ui.Option(label="option 1", position=(10, 10))

    npt.assert_equal(option_test.checked, False)

    if interactive:
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(option_test)
        showm.start()


def test_ui_checkbox_initial_state(interactive=False):
    filename = "test_ui_checkbox"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    checkbox_test = ui.Checkbox(labels=["option 1", "option 2\nOption 2",
                                        "option 3", "option 4"],
                                position=(100, 100),
                                checked_labels=["option 1", "option 4"])

    # Collect the sequence of options that have been checked in this list.
    selected_options = []

    def _on_change(checkbox):
        selected_options.append(list(checkbox.checked_labels))

    # Set up a callback when selection changes
    checkbox_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(checkbox_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY Checkbox")
    show_manager.scene.add(checkbox_test)
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Recorded events:
    #  1. Click on button of option 1.
    #  2. Click on button of option 2.
    #  3. Click on button of option 1.
    #  4. Click on text of option 3.
    #  5. Click on text of option 1.
    #  6. Click on button of option 4.
    #  7. Click on text of option 1.
    #  8. Click on text of option 2.
    #  9. Click on text of option 4.
    #  10. Click on button of option 3.
    # Check if the right options were selected.
    expected = [['option 4'], ['option 4', 'option 2\nOption 2'],
                ['option 4', 'option 2\nOption 2', 'option 1'],
                ['option 4', 'option 2\nOption 2', 'option 1', 'option 3'],
                ['option 4', 'option 2\nOption 2', 'option 3'],
                ['option 2\nOption 2', 'option 3'],
                ['option 2\nOption 2', 'option 3', 'option 1'],
                ['option 3', 'option 1'], ['option 3', 'option 1', 'option 4'],
                ['option 1', 'option 4']]

    npt.assert_equal(len(selected_options), len(expected))
    assert_arrays_equal(selected_options, expected)
    del show_manager

    if interactive:
        checkbox_test = ui.Checkbox(labels=["option 1", "option 2\nOption 2",
                                            "option 3", "option 4"],
                                    position=(100, 100), checked_labels=[])
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(checkbox_test)
        showm.start()


def test_ui_checkbox_default(interactive=False):
    filename = "test_ui_checkbox"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    checkbox_test = ui.Checkbox(labels=["option 1", "option 2\nOption 2",
                                        "option 3", "option 4"],
                                position=(10, 10), checked_labels=[])

    old_positions = []
    for option in checkbox_test.options.values():
        old_positions.append(option.position)

    old_positions = np.asarray(old_positions)
    checkbox_test.position = (100, 100)
    new_positions = []
    for option in checkbox_test.options.values():
        new_positions.append(option.position)
    new_positions = np.asarray(new_positions)
    npt.assert_allclose(new_positions - old_positions,
                        90.0 * np.ones((4, 2)))

    # Collect the sequence of options that have been checked in this list.
    selected_options = []

    def _on_change(checkbox):
        selected_options.append(list(checkbox.checked_labels))

    # Set up a callback when selection changes
    checkbox_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(checkbox_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY Checkbox")
    show_manager.scene.add(checkbox_test)

    # Recorded events:
    #  1. Click on button of option 1.
    #  2. Click on button of option 2.
    #  3. Click on button of option 1.
    #  4. Click on text of option 3.
    #  5. Click on text of option 1.
    #  6. Click on button of option 4.
    #  7. Click on text of option 1.
    #  8. Click on text of option 2.
    #  9. Click on text of option 4.
    #  10. Click on button of option 3.
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right options were selected.
    expected = [['option 1'], ['option 1', 'option 2\nOption 2'],
                ['option 2\nOption 2'], ['option 2\nOption 2', 'option 3'],
                ['option 2\nOption 2', 'option 3', 'option 1'],
                ['option 2\nOption 2', 'option 3', 'option 1', 'option 4'],
                ['option 2\nOption 2', 'option 3', 'option 4'],
                ['option 3', 'option 4'], ['option 3'], []]
    npt.assert_equal(len(selected_options), len(expected))
    assert_arrays_equal(selected_options, expected)
    del show_manager

    if interactive:
        checkbox_test = ui.Checkbox(labels=["option 1", "option 2\nOption 2",
                                            "option 3", "option 4"],
                                    position=(100, 100), checked_labels=[])
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(checkbox_test)
        showm.start()


def test_ui_radio_button_initial_state(interactive=False):
    filename = "test_ui_radio_button"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    radio_button_test = ui.RadioButton(
        labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
        position=(100, 100), checked_labels=['option 4'])

    selected_option = []

    def _on_change(radio_button):
        selected_option.append(radio_button.checked_labels)

    # Set up a callback when selection changes
    radio_button_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(radio_button_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY Checkbox")
    show_manager.scene.add(radio_button_test)

    # Recorded events:
    #  1. Click on button of option 1.
    #  2. Click on button of option 2.
    #  3. Click on button of option 2.
    #  4. Click on text of option 2.
    #  5. Click on button of option 1.
    #  6. Click on text of option 3.
    #  7. Click on button of option 4.
    #  8. Click on text of option 4.
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right options were selected.
    expected = [['option 1'], ['option 2\nOption 2'], ['option 2\nOption 2'],
                ['option 2\nOption 2'], ['option 1'], ['option 3'],
                ['option 4'], ['option 4']]
    npt.assert_equal(len(selected_option), len(expected))
    assert_arrays_equal(selected_option, expected)
    del show_manager

    if interactive:
        radio_button_test = ui.RadioButton(
            labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
            position=(100, 100), checked_labels=[])
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(radio_button_test)
        showm.start()


def test_ui_radio_button_default(interactive=False):
    filename = "test_ui_radio_button"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    radio_button_test = ui.RadioButton(
        labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
        position=(10, 10), checked_labels=[])

    old_positions = []
    for option in radio_button_test.options.values():
        old_positions.append(option.position)
    old_positions = np.asarray(old_positions)
    radio_button_test.position = (100, 100)
    new_positions = []
    for option in radio_button_test.options.values():
        new_positions.append(option.position)
    new_positions = np.asarray(new_positions)
    npt.assert_allclose(new_positions - old_positions,
                        90 * np.ones((4, 2)))

    selected_option = []

    def _on_change(radio_button):
        selected_option.append(radio_button.checked_labels)

    # Set up a callback when selection changes
    radio_button_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(radio_button_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY Checkbox")
    show_manager.scene.add(radio_button_test)

    # Recorded events:
    #  1. Click on button of option 1.
    #  2. Click on button of option 2.
    #  3. Click on button of option 2.
    #  4. Click on text of option 2.
    #  5. Click on button of option 1.
    #  6. Click on text of option 3.
    #  7. Click on button of option 4.
    #  8. Click on text of option 4.
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right options were selected.
    expected = [['option 1'], ['option 2\nOption 2'], ['option 2\nOption 2'],
                ['option 2\nOption 2'], ['option 1'], ['option 3'],
                ['option 4'], ['option 4']]
    npt.assert_equal(len(selected_option), len(expected))
    assert_arrays_equal(selected_option, expected)
    del show_manager

    if interactive:
        radio_button_test = ui.RadioButton(
            labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
            position=(100, 100), checked_labels=[])
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(radio_button_test)
        showm.start()


def test_multiple_radio_button_pre_selected():
    npt.assert_raises(ValueError,
                      ui.RadioButton,
                      labels=["option 1", "option 2\nOption 2",
                              "option 3", "option 4"],
                      checked_labels=["option 1", "option 4"])


def test_ui_listbox_2d(interactive=False):
    filename = "test_ui_listbox_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    # Values that will be displayed by the listbox.
    values = list(range(1, 42 + 1))
    values.append("A Very Very Long Item To Test Text Overflow of List Box 2D")

    if interactive:
        listbox = ui.ListBox2D(values=values,
                               size=(500, 500),
                               multiselection=True,
                               reverse_scrolling=False,
                               background_opacity=0.3)
        listbox.center = (300, 300)
        listbox.panel.opacity = 0.2

        show_manager = window.ShowManager(size=(600, 600),
                                          title="FURY ListBox")
        show_manager.initialize()
        show_manager.scene.add(listbox)
        show_manager.start()

    # Recorded events:
    #  1. Click on 1
    #  2. Ctrl + click on 2,
    #  3. Ctrl + click on 2.
    #  4. Use scroll bar to scroll to the bottom.
    #  5. Click on "A Very Very Long Item...".
    #  6. Use scroll bar to scroll to the top.
    #  7. Click on 1
    #  8. Use mouse wheel to scroll down.
    #  9. Shift + click on "A Very Very Long Item...".
    # 10. Use mouse wheel to scroll back up.

    listbox = ui.ListBox2D(values=values,
                           size=(500, 500),
                           multiselection=True,
                           reverse_scrolling=False)
    listbox.center = (300, 300)

    # We will collect the sequence of values that have been selected.
    selected_values = []

    def _on_change():
        selected_values.append(list(listbox.selected))

    # Set up a callback when selection changes.
    listbox.on_change = _on_change

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(listbox)

    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY ListBox")
    show_manager.scene.add(listbox)
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right values were selected.
    expected = [[1], [1, 2], [1], ["A Very Very Long Item To \
Test Text Overflow of List Box 2D"], [1], values]
    npt.assert_equal(len(selected_values), len(expected))
    assert_arrays_equal(selected_values, expected)

    # Test without multiselection enabled.
    listbox.multiselection = False
    del selected_values[:]  # Clear the list.
    show_manager.play_events_from_file(recording_filename)

    # Check if the right values were selected.
    expected = [[1], [2], [2], ["A Very Very Long Item To \
Test Text Overflow of List Box 2D"], [1], ["A Very Very Long Item To Test \
Text Overflow of List Box 2D"]]
    npt.assert_equal(len(selected_values), len(expected))
    assert_arrays_equal(selected_values, expected)


def test_ui_file_menu_2d(interactive=False):
    filename = "test_ui_file_menu_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    # Create temporary directory and files
    os.mkdir(os.path.join(os.getcwd(), "testdir"))
    os.chdir("testdir")
    os.mkdir(os.path.join(os.getcwd(), "tempdir"))
    for i in range(10):
        open(os.path.join(os.getcwd(), "tempdir", "test" + str(i) + ".txt"),
             'wt').close()
    open("testfile.txt", 'wt').close()

    filemenu = ui.FileMenu2D(size=(500, 500), extensions=["txt"],
                             directory_path=os.getcwd())

    # We will collect the sequence of files that have been selected.
    selected_files = []

    def _on_change():
        selected_files.append(list(filemenu.listbox.selected))

    # Set up a callback when selection changes.
    filemenu.listbox.on_change = _on_change

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(filemenu)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600),
                                      title="FURY FileMenu")
    show_manager.scene.add(filemenu)

    # Recorded events:
    #  1. Click on 'testfile.txt'
    #  2. Click on 'tempdir/'
    #  3. Click on 'test0.txt'.
    #  4. Shift + Click on 'test6.txt'.
    #  5. Click on '../'.
    #  2. Click on 'testfile.txt'.
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right files were selected.
    expected = [["testfile.txt"], ["tempdir"], ["test0.txt"],
                ["test0.txt", "test1.txt", "test2.txt", "test3.txt",
                 "test4.txt", "test5.txt", "test6.txt"],
                ["../"], ["testfile.txt"]]
    npt.assert_equal(len(selected_files), len(expected))
    assert_arrays_equal(selected_files, expected)

    # Remove temporary directory and files
    os.remove("testfile.txt")
    for i in range(10):
        os.remove(os.path.join(os.getcwd(), "tempdir",
                               "test" + str(i) + ".txt"))
    os.rmdir(os.path.join(os.getcwd(), "tempdir"))
    os.chdir("..")
    os.rmdir("testdir")

    if interactive:
        filemenu = ui.FileMenu2D(size=(500, 500), directory_path=os.getcwd())
        show_manager = window.ShowManager(size=(600, 600),
                                          title="FURY FileMenu")
        show_manager.scene.add(filemenu)
        show_manager.start()


def test_ui_combobox_2d(interactive=False):
    filename = "test_ui_combobox_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    values = ["An Item" + str(i) for i in range(0, 5)]
    new_values = ["An Item5", "An Item6"]

    combobox = ui.ComboBox2D(
        items=values, position=(400, 400), size=(300, 200))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(combobox)

    current_size = (800, 800)
    show_manager = window.ShowManager(
        size=current_size, title="ComboBox UI Example")
    show_manager.scene.add(combobox)

    values.extend(new_values)
    combobox.append_item(*new_values)
    npt.assert_equal(values, combobox.items)

    values.append("An Item7")
    combobox.append_item("An Item7")
    npt.assert_equal(values, combobox.items)

    values.append("An Item8")
    values.append("An Item9")
    combobox.append_item("An Item8", "An Item9")
    npt.assert_equal(values, combobox.items)

    complex_list = [[0], (1, [[2, 3], 4], 5)]
    combobox.append_item(*complex_list)
    values.extend([str(i) for i in range(6)])
    npt.assert_equal(values, combobox.items)

    invalid_item = {"Hello": 1, "World": 2}
    npt.assert_raises(TypeError, combobox.append_item, invalid_item)

    npt.assert_equal(values, combobox.items)
    npt.assert_equal((60, 60), combobox.drop_button_size)
    npt.assert_equal([300, 140], combobox.drop_menu_size)
    npt.assert_equal([300, 200], combobox.size)

    ui.ComboBox2D(items=values, draggable=False)

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    npt.assert_equal("An Item1", combobox.selected_text)
    npt.assert_equal(1, combobox.selected_text_index)

    combobox.resize((450, 300))
    npt.assert_equal((360, 90), combobox.text_block_size)
    npt.assert_equal((90, 90), combobox.drop_button_size)
    npt.assert_equal((450, 210), combobox.drop_menu_size)


def test_frame_rate_and_anti_aliasing():
    """Testing frame rate with/out anti-aliasing"""

    length_ = 200
    multi_samples = 32
    max_peels = 8

    st_x = np.arange(length_)
    st_y = np.sin(np.arange(length_))
    st_z = np.zeros(st_x.shape)
    st = np.zeros((length_, 3))
    st[:, 0] = st_x
    st[:, 1] = st_y
    st[:, 2] = st_z

    all_st = []
    all_st.append(st)
    for i in range(1000):
        all_st.append(st + i * np.array([0., .5, 0]))

    # st_actor = actor.line(all_st, linewidth=1)
    # TODO: textblock disappears when lod=True
    st_actor = actor.streamtube(all_st, linewidth=0.1, lod=False)

    scene = window.Scene()
    scene.background((1, 1., 1))

    # quick game style antialiasing
    scene.fxaa_on()
    scene.fxaa_off()

    # the good staff is later with multi-sampling

    tb = ui.TextBlock2D(font_size=40, color=(1, 0.5, 0))

    panel = ui.Panel2D(position=(400, 400), size=(400, 400))
    panel.add_element(tb, (0.2, 0.5))

    counter = itertools.count()
    showm = window.ShowManager(scene,
                               size=(1980, 1080), reset_camera=False,
                               order_transparent=True,
                               multi_samples=multi_samples,
                               max_peels=max_peels,
                               occlusion_ratio=0.0)

    showm.initialize()
    scene.add(panel)
    scene.add(st_actor)
    scene.reset_camera_tight()
    scene.zoom(5)

    class FrameRateHolder(object):
        fpss = []

    frh = FrameRateHolder()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        if cnt % 1 == 0:
            fps = np.round(scene.frame_rate, 0)
            frh.fpss.append(fps)
            msg = "FPS " + str(fps) + ' ' + str(cnt)
            tb.message = msg
            showm.render()
        if cnt > 10:
            showm.exit()

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr = window.snapshot(scene, size=(1980, 1080),
                          offscreen=True,
                          order_transparent=True,
                          multi_samples=multi_samples,
                          max_peels=max_peels,
                          occlusion_ratio=0.0)
    assert_greater(np.sum(arr), 0)
    # TODO: check why in osx we have issues in Azure
    if not skip_osx:
        assert_greater(np.median(frh.fpss), 0)

    frh.fpss = []
    counter = itertools.count()
    multi_samples = 0
    showm = window.ShowManager(scene,
                               size=(1980, 1080), reset_camera=False,
                               order_transparent=True,
                               multi_samples=multi_samples,
                               max_peels=max_peels,
                               occlusion_ratio=0.0)

    showm.initialize()
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr2 = window.snapshot(scene, size=(1980, 1080),
                           offscreen=True,
                           order_transparent=True,
                           multi_samples=multi_samples,
                           max_peels=max_peels,
                           occlusion_ratio=0.0)
    assert_greater(np.sum(arr2), 0)
    if not skip_osx:
        assert_greater(np.median(frh.fpss), 0)


@pytest.mark.skipif(skip_win, reason="This test does not work on Windows."
                                     " Need to be introspected")
def test_timer():
    """Testing add a timer and exit window and app from inside timer."""
    xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 50], [300, 0, 0, 100]])
    xyzr2 = np.array([[0, 200, 0, 30], [100, 200, 0, 50], [300, 200, 0, 100]])
    colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.45]])

    scene = window.Scene()

    sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                                radii=xyzr[:, 3])

    vertices, faces = prim_sphere('repulsion724')

    sphere_actor2 = actor.sphere(centers=xyzr2[:, :3], colors=colors[:],
                                 radii=xyzr2[:, 3], vertices=vertices,
                                 faces=faces.astype('i8'))

    scene.add(sphere_actor)
    scene.add(sphere_actor2)

    tb = ui.TextBlock2D()
    counter = itertools.count()
    showm = window.ShowManager(scene,
                               size=(1024, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()
    scene.add(tb)

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count to 10 and exit :" + str(cnt)
        showm.render()
        if cnt > 9:
            showm.exit()

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    arr = window.snapshot(scene, offscreen=True)
    npt.assert_(np.sum(arr) > 0)
