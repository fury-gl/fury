"""Test for components module."""
import itertools
import os
from os.path import join as pjoin
import shutil
from tempfile import TemporaryDirectory as InTemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, ui, window
from fury.data import DATA_DIR
from fury.decorators import skip_osx, skip_win
from fury.primitive import prim_sphere
from fury.testing import (
    EventCounter,
    assert_arrays_equal,
    assert_equal,
    assert_greater,
    assert_greater_equal,
    assert_less_equal,
    assert_not_equal,
    assert_true,
)

# @pytest.mark.skipif(True, reason="Need investigation. Incorrect "
#                                  "number of event for each vtk version")
from fury.ui import PlaybackPanel


def test_ui_textbox(recording=False):
    filename = 'test_ui_textbox'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    print(recording_filename)
    # TextBox
    textbox_test = ui.TextBox2D(height=3, width=10, text='Text')

    another_textbox_test = ui.TextBox2D(height=3, width=10, text='Enter Text')
    another_textbox_test.set_message('Enter Text')

    # Checking whether textbox went out of focus
    is_off_focused = [False]

    def _off_focus(textbox):
        is_off_focused[0] = True

    # Set up a callback when textbox went out of focus
    textbox_test.off_focus = _off_focus

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(textbox_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title='FURY TextBox')

    show_manager.scene.add(textbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    npt.assert_equal(is_off_focused[0], True)


def test_ui_line_slider_2d_horizontal_bottom(recording=False):
    filename = 'test_ui_line_slider_2d_horizontal_bottom'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_slider_2d_test = ui.LineSlider2D(
        initial_value=-2,
        min_value=-5,
        max_value=5,
        orientation='horizontal',
        text_alignment='bottom',
    )
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(
        size=current_size, title='FURY Horizontal Line Slider'
    )

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
    filename = 'test_ui_line_slider_2d_horizontal_top'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_slider_2d_test = ui.LineSlider2D(
        initial_value=-2,
        min_value=-5,
        max_value=5,
        orientation='horizontal',
        text_alignment='top',
    )
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(
        size=current_size, title='FURY Horizontal Line Slider'
    )

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
    filename = 'test_ui_line_slider_2d_vertical_left'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_slider_2d_test = ui.LineSlider2D(
        initial_value=-2,
        min_value=-5,
        max_value=5,
        orientation='vertical',
        text_alignment='left',
    )
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(
        size=current_size, title='FURY Vertical Line Slider'
    )

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
    filename = 'test_ui_line_slider_2d_vertical_right'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_slider_2d_test = ui.LineSlider2D(
        initial_value=-2,
        min_value=-5,
        max_value=5,
        orientation='vertical',
        text_alignment='right',
    )
    line_slider_2d_test.center = (300, 300)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(
        size=current_size, title='FURY Vertical Line Slider'
    )

    show_manager.scene.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_2d_line_slider_hooks(recording=False):
    global changed, value_changed, slider_moved

    filename = 'test_ui_line_slider_2d_hooks'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_slider_2d = ui.LineSlider2D(center=(300, 300))

    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d)

    show_manager = window.ShowManager(
        size=(600, 600),
        title='FURY Line Slider hooks')

    # counters for the hooks to increment
    changed = value_changed = slider_moved = 0

    def on_line_slider_change(slider):
        global changed
        changed += 1

    def on_line_slider_moved(slider):
        global slider_moved
        slider_moved += 1

    def on_line_slider_value_changed(slider):
        global value_changed
        value_changed += 1

    line_slider_2d.on_change = on_line_slider_change
    line_slider_2d.on_moving_slider = on_line_slider_moved
    line_slider_2d.on_value_changed = on_line_slider_value_changed

    for i in range(100, -1, -1):
        line_slider_2d.value = i

    show_manager.scene.add(line_slider_2d)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    assert_greater(changed, 0)
    assert_greater(value_changed, 0)
    assert_greater(slider_moved, 0)
    assert_equal(changed, value_changed + slider_moved)


def test_ui_line_double_slider_2d(interactive=False):
    line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
        center=(300, 300),
        shape='disk',
        outer_radius=15,
        min_value=-10,
        max_value=10,
        initial_values=(-10, 10),
    )
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.handles[0].size,
        (30, 30))
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.left_disk_value,
        -10
        )
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.right_disk_value,
        10
        )

    line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
        center=(300, 300),
        shape='disk',
        outer_radius=15,
        min_value=-10,
        max_value=10,
        initial_values=(-10, 10),
    )
    npt.assert_equal(
        line_double_slider_2d_vertical_test.handles[0].size,
        (30, 30)
        )
    npt.assert_equal(
        line_double_slider_2d_vertical_test.bottom_disk_value,
        -10)
    npt.assert_equal(
        line_double_slider_2d_vertical_test.top_disk_value,
        10)

    if interactive:
        show_manager = window.ShowManager(
            size=(600, 600), title='FURY Line Double Slider'
        )
        show_manager.scene.add(line_double_slider_2d_horizontal_test)
        show_manager.scene.add(line_double_slider_2d_vertical_test)
        show_manager.start()

    line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
        center=(300, 300),
        shape='square',
        handle_side=5,
        orientation='horizontal',
        initial_values=(50, 40),
    )
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.handles[0].size,
        (5, 5))
    npt.assert_equal(line_double_slider_2d_horizontal_test.left_disk_value, 39)
    npt.assert_equal(
        line_double_slider_2d_horizontal_test.right_disk_value,
        40)
    npt.assert_equal(line_double_slider_2d_horizontal_test.left_disk_ratio, 0.39)
    npt.assert_equal(line_double_slider_2d_horizontal_test.right_disk_ratio, 0.4)

    line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
        center=(300, 300),
        shape='square',
        handle_side=5,
        orientation='vertical',
        initial_values=(50, 40),
    )
    npt.assert_equal(line_double_slider_2d_vertical_test.handles[0].size, (5, 5))
    npt.assert_equal(line_double_slider_2d_vertical_test.bottom_disk_value, 39)
    npt.assert_equal(line_double_slider_2d_vertical_test.top_disk_value, 40)
    npt.assert_equal(line_double_slider_2d_vertical_test.bottom_disk_ratio, 0.39)
    npt.assert_equal(line_double_slider_2d_vertical_test.top_disk_ratio, 0.4)

    with npt.assert_raises(ValueError):
        ui.LineDoubleSlider2D(orientation='Not_hor_not_vert')

    if interactive:
        show_manager = window.ShowManager(
            size=(600, 600), title='FURY Line Double Slider'
        )
        show_manager.scene.add(line_double_slider_2d_horizontal_test)
        show_manager.scene.add(line_double_slider_2d_vertical_test)
        show_manager.start()


def test_ui_2d_line_double_slider_hooks(recording=False):
    global changed, value_changed, slider_moved

    filename = 'test_ui_line_double_slider_2d_hooks'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    line_double_slider_2d = ui.LineDoubleSlider2D(center=(300, 300))

    event_counter = EventCounter()
    event_counter.monitor(line_double_slider_2d)

    show_manager = window.ShowManager(
        size=(600, 600), title='FURY Line Double Slider hooks'
    )

    # counters for the line double slider's changes
    changed = value_changed = slider_moved = 0

    def on_line_double_slider_change(slider):
        global changed
        changed += 1

    def on_line_double_slider_moved(slider):
        global slider_moved
        slider_moved += 1

    def on_line_double_slider_value_changed(slider):
        global value_changed
        value_changed += 1

    line_double_slider_2d.on_change = on_line_double_slider_change
    line_double_slider_2d.on_moving_slider = on_line_double_slider_moved
    line_double_slider_2d.on_value_changed = on_line_double_slider_value_changed

    for i in range(50, -1, -1):
        line_double_slider_2d.left_disk_value = i
        line_double_slider_2d.right_disk_value = 100 - i

    show_manager.scene.add(line_double_slider_2d)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    assert_greater(changed, 0)
    assert_greater(value_changed, 0)
    assert_greater(slider_moved, 0)
    assert_equal(changed, value_changed + slider_moved)


def test_ui_ring_slider_2d(recording=False):
    filename = 'test_ui_ring_slider_2d'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    ring_slider_2d_test = ui.RingSlider2D()
    ring_slider_2d_test.center = (300, 300)
    ring_slider_2d_test.value = 90

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(ring_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title='FURY Ring Slider')

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


def test_ui_2d_ring_slider_hooks(recording=False):
    global changed, value_changed, slider_moved

    filename = 'test_ui_ring_slider_2d_hooks'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    ring_slider_2d = ui.RingSlider2D(center=(300, 300))

    event_counter = EventCounter()
    event_counter.monitor(ring_slider_2d)

    show_manager = window.ShowManager(size=(600, 600), title='FURY Ring Slider hooks')

    # counters for the ring slider changes
    changed = value_changed = slider_moved = 0

    def on_ring_slider_change(slider):
        global changed
        changed += 1

    def on_ring_slider_moved(slider):
        global slider_moved
        slider_moved += 1

    def on_ring_slider_value_changed(slider):
        global value_changed
        value_changed += 1

    ring_slider_2d.on_change = on_ring_slider_change
    ring_slider_2d.on_moving_slider = on_ring_slider_moved
    ring_slider_2d.on_value_changed = on_ring_slider_value_changed

    for i in range(360, -1, -1):
        ring_slider_2d.value = i

    show_manager.scene.add(ring_slider_2d)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    assert_greater(changed, 0)
    assert_greater(value_changed, 0)
    assert_greater(slider_moved, 0)
    assert_equal(changed, value_changed + slider_moved)


def test_ui_range_slider(interactive=False):
    range_slider_test_horizontal = ui.RangeSlider(shape='square')
    range_slider_test_vertical = ui.RangeSlider(shape='square', orientation='vertical')

    if interactive:
        show_manager = window.ShowManager(
            size=(600, 600), title='FURY Line Double Slider'
        )
        show_manager.scene.add(range_slider_test_horizontal)
        show_manager.scene.add(range_slider_test_vertical)
        show_manager.start()


def test_ui_slider_value_range():
    with npt.assert_no_warnings():
        # LineSlider2D
        line_slider = ui.LineSlider2D(min_value=0, max_value=0)
        assert_equal(line_slider.value, 0)
        assert_equal(line_slider.min_value, 0)
        assert_equal(line_slider.max_value, 0)
        line_slider.value = 100
        assert_equal(line_slider.value, 0)
        line_slider.value = -100
        assert_equal(line_slider.value, 0)

        line_slider = ui.LineSlider2D(min_value=0, max_value=100)
        line_slider.value = 105
        assert_equal(line_slider.value, 100)
        line_slider.value = -100
        assert_equal(line_slider.value, 0)

        # LineDoubleSlider2D
        line_double_slider = ui.LineDoubleSlider2D(min_value=0, max_value=0)
        assert_equal(line_double_slider.left_disk_value, 0)
        assert_equal(line_double_slider.right_disk_value, 0)
        line_double_slider.left_disk_value = 100
        assert_equal(line_double_slider.left_disk_value, 0)
        line_double_slider.right_disk_value = -100
        assert_equal(line_double_slider.right_disk_value, 0)

        line_double_slider = ui.LineDoubleSlider2D(min_value=50, max_value=100)
        line_double_slider.right_disk_value = 150
        assert_equal(line_double_slider.right_disk_value, 100)
        line_double_slider.left_disk_value = -150
        assert_equal(line_double_slider.left_disk_value, 50)

        # RingSlider2D
        ring_slider = ui.RingSlider2D(initial_value=0, min_value=0, max_value=0)
        assert_equal(ring_slider.value, 0)
        assert_equal(ring_slider.previous_value, 0)
        ring_slider.value = 180
        assert_equal(ring_slider.value, 0)
        ring_slider.value = -180
        assert_equal(ring_slider.value, 0)

        # RangeSlider
        range_slider_2d = ui.RangeSlider(min_value=0, max_value=0)
        assert_equal(range_slider_2d.value_slider.value, 0)
        range_slider_2d.value_slider.value = 100
        assert_equal(range_slider_2d.value_slider.value, 0)


def test_ui_option(interactive=False):
    option_test = ui.Option(label='option 1', position=(10, 10))

    npt.assert_equal(option_test.checked, False)

    if interactive:
        showm = window.ShowManager(size=(600, 600))
        showm.scene.add(option_test)
        showm.start()


def test_ui_checkbox_initial_state(recording=False):
    filename = 'test_ui_checkbox_initial_state'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    checkbox_test = ui.Checkbox(
        labels=['option 1', 'option 2\nOption 2', 'option 3', 'option 4'],
        position=(100, 100),
        checked_labels=['option 1', 'option 4'],
    )

    # Collect the sequence of options that have been checked in this list.
    selected_options = []

    def _on_change(checkbox):
        selected_options.append(list(checkbox.checked_labels))

    # Set up a callback when selection changes
    checkbox_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(checkbox_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600), title='FURY Checkbox')
    show_manager.scene.add(checkbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
        print(selected_options)
    else:
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
        expected = [
            ['option 4'],
            ['option 4', 'option 2\nOption 2'],
            ['option 4', 'option 2\nOption 2', 'option 1'],
            ['option 4', 'option 2\nOption 2', 'option 1', 'option 3'],
            ['option 4', 'option 2\nOption 2', 'option 3'],
            ['option 2\nOption 2', 'option 3'],
            ['option 2\nOption 2', 'option 3', 'option 1'],
            ['option 3', 'option 1'],
            ['option 3', 'option 1', 'option 4'],
            ['option 1', 'option 4'],
        ]

        npt.assert_equal(len(selected_options), len(expected))
        assert_arrays_equal(selected_options, expected)


def test_ui_checkbox_default(recording=False):
    filename = 'test_ui_checkbox_initial_state'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    checkbox_test = ui.Checkbox(
        labels=['option 1', 'option 2\nOption 2', 'option 3', 'option 4'],
        position=(10, 10),
        checked_labels=[],
    )

    old_positions = []
    for option in checkbox_test.options.values():
        old_positions.append(option.position)

    old_positions = np.asarray(old_positions)
    checkbox_test.position = (100, 100)
    new_positions = []
    for option in checkbox_test.options.values():
        new_positions.append(option.position)
    new_positions = np.asarray(new_positions)
    npt.assert_allclose(new_positions - old_positions, 90.0 * np.ones((4, 2)))

    # Collect the sequence of options that have been checked in this list.
    selected_options = []

    def _on_change(checkbox):
        selected_options.append(list(checkbox.checked_labels))

    # Set up a callback when selection changes
    checkbox_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(checkbox_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600), title='FURY Checkbox')
    show_manager.scene.add(checkbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
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
        expected = [
            ['option 1'],
            ['option 1', 'option 2\nOption 2'],
            ['option 2\nOption 2'],
            ['option 2\nOption 2', 'option 3'],
            ['option 2\nOption 2', 'option 3', 'option 1'],
            ['option 2\nOption 2', 'option 3', 'option 1', 'option 4'],
            ['option 2\nOption 2', 'option 3', 'option 4'],
            ['option 3', 'option 4'],
            ['option 3'],
            [],
        ]
        npt.assert_equal(len(selected_options), len(expected))
        assert_arrays_equal(selected_options, expected)


def test_ui_radio_button_initial_state(recording=False):
    filename = 'test_ui_radio_button_initial'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    radio_button_test = ui.RadioButton(
        labels=['option 1', 'option 2\nOption 2', 'option 3', 'option 4'],
        position=(100, 100),
        checked_labels=['option 4'],
    )

    selected_option = []

    def _on_change(radio_button):
        selected_option.append(radio_button.checked_labels)

    # Set up a callback when selection changes
    radio_button_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(radio_button_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600), title='FURY Checkbox')
    show_manager.scene.add(radio_button_test)
    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
    else:
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
        expected = [
            ['option 1'],
            ['option 2\nOption 2'],
            ['option 2\nOption 2'],
            ['option 2\nOption 2'],
            ['option 1'],
            ['option 3'],
            ['option 4'],
            ['option 4'],
        ]
        npt.assert_equal(len(selected_option), len(expected))
        assert_arrays_equal(selected_option, expected)


def test_ui_radio_button_default(recording=False):
    filename = 'test_ui_radio_button_initial'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    radio_button_test = ui.RadioButton(
        labels=['option 1', 'option 2\nOption 2', 'option 3', 'option 4'],
        position=(10, 10),
        checked_labels=[],
    )

    old_positions = []
    for option in radio_button_test.options.values():
        old_positions.append(option.position)
    old_positions = np.asarray(old_positions)
    radio_button_test.position = (100, 100)
    new_positions = []
    for option in radio_button_test.options.values():
        new_positions.append(option.position)
    new_positions = np.asarray(new_positions)
    npt.assert_allclose(new_positions - old_positions, 90 * np.ones((4, 2)))

    selected_option = []

    def _on_change(radio_button):
        selected_option.append(radio_button.checked_labels)

    # Set up a callback when selection changes
    radio_button_test.on_change = _on_change

    event_counter = EventCounter()
    event_counter.monitor(radio_button_test)

    # Create a show manager and record/play events.
    show_manager = window.ShowManager(size=(600, 600), title='FURY Checkbox')
    show_manager.scene.add(radio_button_test)
    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
    else:
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
        expected = [
            ['option 1'],
            ['option 2\nOption 2'],
            ['option 2\nOption 2'],
            ['option 2\nOption 2'],
            ['option 1'],
            ['option 3'],
            ['option 4'],
            ['option 4'],
        ]
        npt.assert_equal(len(selected_option), len(expected))
        assert_arrays_equal(selected_option, expected)


def test_multiple_radio_button_pre_selected():
    npt.assert_raises(
        ValueError,
        ui.RadioButton,
        labels=['option 1', 'option 2\nOption 2', 'option 3', 'option 4'],
        checked_labels=['option 1', 'option 4'],
    )


@pytest.mark.skipif(
    True, reason='Need investigation. Incorrect ' 'number of event for each vtk version'
)
def test_ui_listbox_2d(interactive=False):
    filename = 'test_ui_listbox_2d'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    # Values that will be displayed by the listbox.
    values = list(range(1, 42 + 1))
    values.append('A Very Very Long Item To Test Text Overflow of List Box 2D')

    if interactive:
        listbox = ui.ListBox2D(
            values=values,
            size=(500, 500),
            multiselection=True,
            reverse_scrolling=False,
            background_opacity=0.3,
        )
        listbox.center = (300, 300)
        listbox.panel.opacity = 0.2

        show_manager = window.ShowManager(size=(600, 600), title='FURY ListBox')
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

    listbox = ui.ListBox2D(
        values=values, size=(500, 500), multiselection=True, reverse_scrolling=False
    )
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

    show_manager = window.ShowManager(size=(600, 600), title='FURY ListBox')
    show_manager.scene.add(listbox)
    show_manager.play_events_from_file(recording_filename)
    expected = EventCounter.load(expected_events_counts_filename)
    event_counter.check_counts(expected)

    # Check if the right values were selected.
    expected = [
        [1],
        [1, 2],
        [1],
        [
            'A Very Very Long Item To \
Test Text Overflow of List Box 2D'
        ],
        [1],
        values,
    ]
    npt.assert_equal(len(selected_values), len(expected))
    assert_arrays_equal(selected_values, expected)

    # Test without multiselection enabled.
    listbox.multiselection = False
    del selected_values[:]  # Clear the list.
    show_manager.play_events_from_file(recording_filename)

    # Check if the right values were selected.
    expected = [
        [1],
        [2],
        [2],
        [
            'A Very Very Long Item To \
Test Text Overflow of List Box 2D'
        ],
        [1],
        [
            'A Very Very Long Item To Test \
Text Overflow of List Box 2D'
        ],
    ]
    npt.assert_equal(len(selected_values), len(expected))
    assert_arrays_equal(selected_values, expected)


def test_ui_listbox_2d_visibility():
    l1 = ui.ListBox2D(
        values=['Violet', 'Indigo', 'Blue', 'Yellow'],
        position=(12, 10),
        size=(100, 100),
    )
    l2 = ui.ListBox2D(
        values=['Violet', 'Indigo', 'Blue', 'Yellow'],
        position=(10, 10),
        size=(100, 300),
    )

    def assert_listbox(list_box, expected_scroll_bar_height):
        view_end = list_box.view_offset + list_box.nb_slots
        assert list_box.scroll_bar.height == expected_scroll_bar_height
        for slot in list_box.slots[view_end:]:
            assert slot.size[1] == list_box.slot_height

    assert_listbox(l1, 40.0)

    # Assert that for list 2 the slots and scrollbars aren't visible.
    assert_listbox(l2, 0)


def test_ui_file_menu_2d(interactive=False):
    filename = 'test_ui_file_menu_2d'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    with InTemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, 'testdir')
        os.makedirs(os.path.join(test_dir, 'tempdir'))
        for i in range(10):
            open(os.path.join(test_dir, 'tempdir', f'test{i}.txt'), 'wt').close()
        open(os.path.join(test_dir, 'testfile.txt'), 'wt').close()

        filemenu = ui.FileMenu2D(
            size=(500, 500), extensions=['txt'], directory_path=test_dir
        )

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
        show_manager = window.ShowManager(size=(600, 600), title='FURY FileMenu')
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
        expected = [
            ['testfile.txt'],
            ['tempdir'],
            ['test0.txt'],
            [
                'test0.txt',
                'test1.txt',
                'test2.txt',
                'test3.txt',
                'test4.txt',
                'test5.txt',
                'test6.txt',
            ],
            ['../'],
            ['testfile.txt'],
        ]

        npt.assert_equal(len(selected_files), len(expected))
        assert_arrays_equal(selected_files, expected)
        if interactive:
            filemenu = ui.FileMenu2D(size=(500, 500), directory_path=os.getcwd())
            show_manager = window.ShowManager(size=(600, 600), title='FURY FileMenu')
            show_manager.scene.add(filemenu)
            show_manager.start()


def test_ui_combobox_2d(interactive=False):
    filename = 'test_ui_combobox_2d'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    values = ['An Item' + str(i) for i in range(0, 5)]
    new_values = ['An Item5', 'An Item6']

    combobox = ui.ComboBox2D(items=values, position=(400, 400), size=(300, 200))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(combobox)

    current_size = (800, 800)
    show_manager = window.ShowManager(size=current_size, title='ComboBox UI Example')
    show_manager.scene.add(combobox)

    values.extend(new_values)
    combobox.append_item(*new_values)
    npt.assert_equal(values, combobox.items)

    values.append('An Item7')
    combobox.append_item('An Item7')
    npt.assert_equal(values, combobox.items)

    values.append('An Item8')
    values.append('An Item9')
    combobox.append_item('An Item8', 'An Item9')
    npt.assert_equal(values, combobox.items)

    complex_list = [[0], (1, [[2, 3], 4], 5)]
    combobox.append_item(*complex_list)
    values.extend([str(i) for i in range(6)])
    npt.assert_equal(values, combobox.items)

    invalid_item = {'Hello': 1, 'World': 2}
    npt.assert_raises(TypeError, combobox.append_item, invalid_item)

    npt.assert_equal(values, combobox.items)
    npt.assert_equal((30, 20), combobox.drop_button_size)
    npt.assert_equal([270, 140], combobox.drop_menu_size)
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

    npt.assert_equal('An Item1', combobox.selected_text)
    npt.assert_equal(1, combobox.selected_text_index)

    combobox.resize((450, 300))
    npt.assert_equal((405, 30), combobox.text_block_size)
    npt.assert_equal((45, 30), combobox.drop_button_size)
    npt.assert_equal((405, 210), combobox.drop_menu_size)


def test_ui_combobox_2d_dropdown_visibility(interactive=False):

    values = ['An Item' + str(i) for i in range(0, 5)]

    tab_ui = ui.TabUI(position=(49, 94), size=(400, 400), nb_tabs=1, draggable=True)
    combobox = ui.ComboBox2D(
        items=values,
        position=(400, 400), size=(300, 200)
        )

    tab_ui.add_element(0, combobox, (0.1, 0.3))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(combobox)
    event_counter.monitor(tab_ui)

    current_size = (800, 800)
    show_manager = window.ShowManager(size=current_size, title='ComboBox UI Example')
    show_manager.scene.add(tab_ui)

    tab_ui.tabs[0].content_panel.set_visibility(True)
    npt.assert_equal(False, combobox._menu_visibility)
    npt.assert_equal(
        False,
        combobox.drop_down_menu.panel.actors[0].GetVisibility()
        )
    npt.assert_equal(0, combobox.drop_down_button.current_icon_id)
    npt.assert_equal(True, combobox.drop_down_button.actors[0].GetVisibility())
    npt.assert_equal(True, combobox.selection_box.actors[0].GetVisibility())

    tab_ui.tabs[0].content_panel.set_visibility(False)
    npt.assert_equal(False, combobox._menu_visibility)
    npt.assert_equal(
        False,
        combobox.drop_down_menu.panel.actors[0].GetVisibility()
        )
    npt.assert_equal(0, combobox.drop_down_button.current_icon_id)
    npt.assert_equal(
        False,
        combobox.drop_down_button.actors[0].GetVisibility()
        )
    npt.assert_equal(False, combobox.selection_box.actors[0].GetVisibility())

    iren = show_manager.scene.GetRenderWindow().GetInteractor().GetInteractorStyle()
    combobox.menu_toggle_callback(iren, None, None)
    tab_ui.tabs[0].content_panel.set_visibility(True)
    npt.assert_equal(True, combobox._menu_visibility)
    npt.assert_equal(
        True,
        combobox.drop_down_menu.panel.actors[0].GetVisibility()
        )
    npt.assert_equal(1, combobox.drop_down_button.current_icon_id)
    npt.assert_equal(True, combobox.drop_down_button.actors[0].GetVisibility())
    npt.assert_equal(True, combobox.selection_box.actors[0].GetVisibility())


@pytest.mark.skipif(
    skip_osx,
    reason='This test does not work on macOS.'
    'It works on the local machines.'
    'The colors provided for shapes are '
    'normalized values whereas when we test'
    'it, the values returned are between '
    '0-255. So while conversion from one'
    'representation to another, there may be'
    'something which causes these issues.',
)
def test_ui_draw_shape():
    line = ui.DrawShape(shape_type='line', position=(150, 150))
    quad = ui.DrawShape(shape_type='quad', position=(300, 300))
    circle = ui.DrawShape(shape_type='circle', position=(150, 300))

    with npt.assert_raises(IOError):
        ui.DrawShape('poly')

    line.resize((100, 5))
    line.shape.color = (0, 1, 0)
    quad.resize((150, 150))
    quad.shape.color = (1, 0, 0)
    circle.resize((25, 0))
    circle.shape.color = (0, 0, 1)

    line_color = np.round(255 * np.array(line.shape.color)).astype('uint8')
    quad_color = np.round(255 * np.array(quad.shape.color)).astype('uint8')
    circle_color = np.round(255 * np.array(circle.shape.color)).astype('uint8')

    current_size = (900, 900)
    scene = window.Scene()
    show_manager = window.ShowManager(
        scene, size=current_size, title='DrawShape UI Example'
    )
    scene.add(line, circle, quad)

    arr = window.snapshot(show_manager.scene, size=(800, 800))
    report = window.analyze_snapshot(
        arr, colors=[tuple(line_color), tuple(circle_color), tuple(quad_color)]
    )
    npt.assert_equal(report.objects, 3)
    npt.assert_equal(report.colors_found, [True, True, True])


def test_ui_draw_panel_basic(interactive=False):
    filename = 'test_ui_draw_panel_basic'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    drawpanel = ui.DrawPanel(size=(600, 600), position=(30, 10))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(drawpanel)

    current_size = (680, 680)
    show_manager = window.ShowManager(
        size=current_size, title='DrawPanel Basic UI Example'
    )
    show_manager.scene.add(drawpanel)

    # Recorded events:
    #  1. Check all mode selection button
    #  2. Creation and clamping of shapes
    #  3. Transformation and clamping of shapes

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_draw_panel_rotation(interactive=False):
    filename = 'test_ui_draw_panel_rotation'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    drawpanel = ui.DrawPanel(size=(600, 600), position=(30, 10))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(drawpanel)

    current_size = (680, 680)
    show_manager = window.ShowManager(
        size=current_size, title='DrawPanel Rotation UI Example'
    )
    show_manager.scene.add(drawpanel)

    # Recorded events:
    #  1. Rotation and clamping of shape

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_playback_panel(interactive=False):
    global playing, paused, stopped, loop, ts

    playing = stopped = paused = loop = False
    ts = 0

    current_size = (900, 620)
    show_manager = window.ShowManager(
        size=current_size, title='PlaybackPanel UI Example'
    )

    filename = 'test_playback_panel'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    def play():
        global playing
        playing = True

    def pause():
        global paused
        paused = True

    def stop():
        global stopped
        stopped = True

    def loop_toggle(value):
        global loop
        loop = True

    def change_t(value):
        global ts
        ts = value
        assert_greater_equal(playback.current_time, 0)
        assert_less_equal(playback.current_time, playback.final_time)
        assert_equal(playback.current_time, ts)

    playback = PlaybackPanel()
    playback.on_play = play
    playback.on_pause = pause
    playback.on_stop = stop
    playback.on_loop_toggle = loop_toggle
    playback.on_progress_bar_changed = change_t

    show_manager.scene.add(playback)
    event_counter = EventCounter()
    event_counter.monitor(playback)

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    assert_true(playing)
    assert_true(paused)
    assert_true(stopped)
    assert_equal(playback.current_time, ts)
    assert_greater(playback.current_time, 0)
    assert_not_equal(playback.current_time_str, '00:00.00')
    playback.current_time = 5
    assert_equal(playback.current_time, 5)
    assert_equal(playback.current_time_str, '00:05.00')
    # test show/hide
    playback.show()
    ss = window.snapshot(show_manager.scene)
    assert_not_equal(np.max(ss), 0)
    playback.hide()
    ss = window.snapshot(show_manager.scene)
    assert_equal(np.max(ss), 0)


def test_card_ui(interactive=False):
    filename = 'test_card_ui'
    recording_filename = pjoin(DATA_DIR, filename + '.log.gz')
    expected_events_counts_filename = pjoin(DATA_DIR, filename + '.json')

    img_url = "https://raw.githubusercontent.com/fury-gl"\
              "/fury-communication-assets/main/fury-logo.png"

    title = "FURY"
    body = "FURY - Free Unified Rendering in pYthon."\
           "A software library for scientific visualization in Python."

    card = ui.elements.Card2D(image_path=img_url, draggable=True,
                              title_text=title, body_text=body,
                              image_scale=0.5)

    # Assign the counter callback to every possible event.

    event_counter = EventCounter()
    event_counter.monitor(card)

    npt.assert_equal(card.size, (400.0, 400.0))
    npt.assert_equal(card.image.size[1], 200.0)
    npt.assert_equal(card.title, title)
    npt.assert_equal(card.body, body)
    npt.assert_equal(card.color, (0.5, 0.5, 0.5))
    npt.assert_equal(card.panel.position, (0, 0))

    card.title = 'Changed Title'
    npt.assert_equal(card.title, 'Changed Title')

    card.body = 'Changed Body'
    npt.assert_equal(card.body, 'Changed Body')

    card.title = title
    card.body = body
    card.color = (1.0, 1.0, 1.0)
    npt.assert_equal(card.color, (1.0, 1.0, 1.0))

    card.resize((300, 300))
    npt.assert_equal(card.image.size[1], 150.0)
    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title='FURY Card')
    show_manager.scene.add(card)

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


def test_ui_spinbox(interactive=False):
    filename = "test_ui_spinbox"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

    spinbox = ui.SpinBox(size=(300, 200), min_val=-20, max_val=10, step=2)
    npt.assert_equal(spinbox.value, 10)

    spinbox.value = 5
    npt.assert_equal(spinbox.value, 5)
    spinbox.value = 50
    npt.assert_equal(spinbox.value, 10)
    spinbox.value = -50
    npt.assert_equal(spinbox.value, -20)

    spinbox.min_val = -100
    spinbox.max_val = 100

    spinbox.value = 5
    npt.assert_equal(spinbox.value, 5)
    spinbox.value = 50
    npt.assert_equal(spinbox.value, 50)
    spinbox.value = -50
    npt.assert_equal(spinbox.value, -50)

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(spinbox)

    current_size = (800, 800)
    show_manager = window.ShowManager(
        size=current_size,
        title="SpinBox UI Example")
    show_manager.scene.add(spinbox)

    if interactive:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)
    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)

    spinbox.resize((450, 200))
    npt.assert_equal((315, 160), spinbox.textbox_size)
    npt.assert_equal((90, 60), spinbox.button_size)
