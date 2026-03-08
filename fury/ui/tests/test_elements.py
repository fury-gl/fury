"""Test for components module."""

from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.data import fetch_viz_icons, read_viz_icons
from fury.ui.helpers import Anchor


def test_textured_button_2d_initialization():
    """Test TexturedButton2D initialization and texture loading."""
    fetch_viz_icons()

    states = {
        "default": read_viz_icons(fname="play3.png"),
        "hover": read_viz_icons(fname="pause2.png"),
        "pressed": read_viz_icons(fname="stop2.png"),
    }

    button_size = (50, 50)
    button_pos = (100, 100)

    button = ui.TexturedButton2D(states=states, position=button_pos, size=button_size)

    npt.assert_equal(button.size, button_size)
    npt.assert_array_equal(button.get_position(Anchor.LEFT, Anchor.TOP), button_pos)
    assert "default" in button.texture_map
    assert "hover" in button.texture_map
    assert button.child is not None


def test_textured_button_state_updates():
    """Test if TexturedButton2D updates its visual state on interaction."""
    fetch_viz_icons()
    states = {
        "default": read_viz_icons(fname="play3.png"),
        "hover": read_viz_icons(fname="pause2.png"),
    }

    button = ui.TexturedButton2D(states=states, size=(40, 40))

    button.update_visual_state()
    initial_tex = button.child.material.map

    button.is_hovered = True
    button.update_visual_state()
    hover_tex = button.child.material.map

    assert initial_tex != hover_tex


def test_text_button_2d_initialization_default():
    """Test TextButton2D with default states and labels."""
    label = "Click Me"
    button = ui.TextButton2D(label=label, size=(120, 50))

    npt.assert_equal(button.default_label, label)
    assert isinstance(button.child, ui.TextBlock2D)
    npt.assert_equal(button.child.message, label)

    npt.assert_equal(button.child.background.size, (120, 50))


def test_text_button_2d_custom_states():
    """Test TextButton2D with complex state dictionary (text + color)."""
    custom_states = {
        "default": {"text": "Idle", "color": (1, 1, 1)},
        "hover": {"text": "Hovering...", "color": (0, 1, 0)},
        "pressed": {"text": "Clicked!", "color": (1, 0, 0)},
    }

    button = ui.TextButton2D(label="Default", states=custom_states)

    button.update_visual_state()
    npt.assert_equal(button.child.message, "Idle")
    npt.assert_array_almost_equal(button.child.background.color, (1, 1, 1))

    button.is_hovered = True
    button.update_visual_state()
    npt.assert_equal(button.child.message, "Hovering...")
    npt.assert_array_almost_equal(button.child.background.color, (0, 1, 0))

    button.is_pressed = True
    button.update_visual_state()
    npt.assert_equal(button.child.message, "Clicked!")
    npt.assert_array_almost_equal(button.child.background.color, (1, 0, 0))


def test_text_button_2d_visual_snapshot():
    """Visual test to ensure TextButton2D background color renders."""
    rect_color = (0.0, 0.0, 1.0)
    states = {"default": rect_color}

    button = ui.TextButton2D(
        label="BlueBtn", states=states, size=(100, 50), position=(50, 50)
    )

    scene = window.Scene()
    scene.add(button)
    button.update_visual_state()

    fname = "text_button_render.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_colors = np.mean(img_array.reshape(-1, img_array.shape[2]), axis=0)
    assert mean_colors[2] > mean_colors[0]
    assert mean_colors[2] > mean_colors[1]


def test_text_button_resize():
    """Test if resizing the button resizes the internal TextBlock."""
    button = ui.TextButton2D(label="ResizeTest", size=(100, 40))
    new_size = (200, 80)

    button.resize(new_size)

    npt.assert_equal(button.size, new_size)
    npt.assert_equal(button.child.background.size, new_size)


def test_line_slider_2d_functional_initialization():
    """Test property assignment and initial state calculation."""
    slider = ui.LineSlider2D(
        initial_value=25,
        min_value=0,
        max_value=100,
        length=250,
        orientation="horizontal",
        shape="square",
        handle_side=30,
    )

    #
    npt.assert_equal(slider.value, 25)
    npt.assert_almost_equal(slider.ratio, 0.25)
    npt.assert_equal(slider.shape, "square")
    npt.assert_equal(slider.orientation, "horizontal")

    assert isinstance(slider.handle, ui.Rectangle2D)
    npt.assert_equal(slider.handle.size, (30, 30))


def test_line_slider_2d_programmatic_clamping():
    """Verify that the value/ratio setters strictly enforce bounds."""
    slider = ui.LineSlider2D(min_value=0, max_value=10, initial_value=5)

    slider.value = 15
    npt.assert_equal(slider.value, 10)
    npt.assert_equal(slider.ratio, 1.0)

    slider.value = -5
    npt.assert_equal(slider.value, 0)
    npt.assert_equal(slider.ratio, 0.0)

    slider.ratio = 2.0
    npt.assert_equal(slider.ratio, 1.0)
    npt.assert_equal(slider.value, 10)


def test_line_slider_2d_synchronization():
    """Test if changing ratio updates value and vice versa."""
    slider = ui.LineSlider2D(min_value=100, max_value=200)

    slider.ratio = 0.5
    npt.assert_equal(slider.value, 150)

    slider.value = 110
    npt.assert_almost_equal(slider.ratio, 0.1)


def test_line_slider_2d_layout_logic():
    """Verify programmatic size and actor placement logic."""
    length = 300
    line_w = 10
    slider = ui.LineSlider2D(length=length, line_width=line_w, orientation="vertical")

    size = slider._get_size()

    npt.assert_equal(size[1], length)

    assert size[0] >= line_w

    slider.ratio = 0.0
    pos_start = slider.handle.get_position().copy()
    slider.ratio = 1.0
    pos_end = slider.handle.get_position().copy()

    assert pos_start[1] != pos_end[1]


def test_line_slider_2d_text_formatting():
    """Test the template system for programmatic text updates."""
    custom_template = "Value: {value:.2f}"
    slider = ui.LineSlider2D(initial_value=5.678, text_template=custom_template)

    assert slider.text.message == "Value: 5.68"

    slider.value = 1.234
    assert slider.text.message == "Value: 1.23"


def test_line_slider_2d_callback_logic():
    """Test that setting values triggers the correct programmatic hooks."""
    slider = ui.LineSlider2D(initial_value=0)

    hooks_triggered = {"value_changed": False}

    def v_callback(u):
        hooks_triggered["value_changed"] = True

    slider.on_value_changed = v_callback

    slider.value = 10
    assert hooks_triggered["value_changed"] is True


def test_line_slider_2d_visibility_propagation():
    """Test if setting visibility on the parent propagates to sub-components."""
    slider = ui.LineSlider2D()

    slider.set_visibility(False)
    for actor in slider._get_actors():
        assert actor.visible is False

    slider.set_visibility(True)
    for actor in slider._get_actors():
        assert actor.visible is True


# --- LineDoubleSlider2D Tests ---


def test_line_double_slider_2d_initialization():
    """Test LineDoubleSlider2D initialization with defaults and variants."""
    slider = ui.LineDoubleSlider2D()
    npt.assert_equal(slider.left_disk_value, 0)
    npt.assert_equal(slider.right_disk_value, 100)
    npt.assert_equal(slider.orientation, "horizontal")
    npt.assert_equal(slider.shape, "disk")
    assert isinstance(slider.handles[0], ui.Disk2D)
    assert isinstance(slider.handles[1], ui.Disk2D)
    assert len(slider.handles) == 2
    assert len(slider.text) == 2

    slider_sq = ui.LineDoubleSlider2D(shape="square", handle_side=15)
    assert isinstance(slider_sq.handles[0], ui.Rectangle2D)
    npt.assert_equal(slider_sq.handles[0].size, (15, 15))

    slider_v = ui.LineDoubleSlider2D(orientation="vertical")
    npt.assert_equal(slider_v.orientation, "vertical")


def test_line_double_slider_2d_clamping():
    """Test that values are clamped to min/max."""
    slider = ui.LineDoubleSlider2D(min_value=10, max_value=90, initial_values=(10, 90))

    slider.left_disk_value = -50
    npt.assert_equal(slider.left_disk_value, 10)

    slider.right_disk_value = 200
    npt.assert_equal(slider.right_disk_value, 90)


def test_line_double_slider_2d_crossing_prevention():
    """Test that handles cannot cross each other."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=100, initial_values=(30, 70))

    slider.left_disk_value = 80
    assert slider.left_disk_value <= slider.right_disk_value

    slider.right_disk_value = 20
    assert slider.right_disk_value >= slider.left_disk_value


def test_line_double_slider_2d_synchronization():
    """Test ratio-value synchronization."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=100, initial_values=(0, 100))

    slider.left_disk_ratio = 0.5
    npt.assert_almost_equal(slider.left_disk_value, 50)

    slider.right_disk_value = 75
    npt.assert_almost_equal(slider.right_disk_ratio, 0.75)


def test_line_double_slider_2d_alias_properties():
    """Test that left/bottom and right/top aliases match."""
    slider = ui.LineDoubleSlider2D(initial_values=(20, 80))

    npt.assert_equal(slider.left_disk_value, slider.bottom_disk_value)
    npt.assert_equal(slider.right_disk_value, slider.top_disk_value)
    npt.assert_equal(slider.left_disk_ratio, slider.bottom_disk_ratio)
    npt.assert_equal(slider.right_disk_ratio, slider.top_disk_ratio)

    slider.bottom_disk_value = 25
    npt.assert_equal(slider.left_disk_value, 25)

    slider.top_disk_ratio = 0.9
    npt.assert_almost_equal(slider.right_disk_ratio, 0.9)


def test_line_double_slider_2d_layout():
    """Test _get_size() for both orientations."""
    slider_h = ui.LineDoubleSlider2D(
        length=300, line_width=10, orientation="horizontal"
    )
    size_h = slider_h._get_size()
    npt.assert_equal(size_h[0], 300)
    assert size_h[1] >= 10

    slider_v = ui.LineDoubleSlider2D(length=300, line_width=10, orientation="vertical")
    size_v = slider_v._get_size()
    npt.assert_equal(size_v[1], 300)
    assert size_v[0] >= 10


def test_line_double_slider_2d_text_formatting():
    """Test string and callable text templates."""
    slider = ui.LineDoubleSlider2D(
        initial_values=(25, 75),
        text_template="{value:.2f}",
    )
    assert slider.text[0].message == "25.00"
    assert slider.text[1].message == "75.00"

    def custom_template(s):
        return "custom"

    slider2 = ui.LineDoubleSlider2D(
        initial_values=(10, 90),
        text_template=custom_template,
    )
    assert slider2.text[0].message == "custom"
    assert slider2.text[1].message == "custom"


def test_line_double_slider_2d_callbacks():
    """Test that on_value_changed fires on value set."""
    slider = ui.LineDoubleSlider2D(initial_values=(0, 100))
    triggered = {"count": 0}

    def on_change(s):
        triggered["count"] += 1

    slider.on_value_changed = on_change

    slider.left_disk_value = 10
    assert triggered["count"] >= 1

    prev = triggered["count"]
    slider.right_disk_value = 90
    assert triggered["count"] > prev


def test_line_double_slider_2d_visibility():
    """Test visibility propagation to all sub-actors."""
    slider = ui.LineDoubleSlider2D()

    slider.set_visibility(False)
    for actor in slider._get_actors():
        assert actor.visible is False

    slider.set_visibility(True)
    for actor in slider._get_actors():
        assert actor.visible is True


def test_line_double_slider_2d_zero_range():
    """Test edge case where min_value equals max_value."""
    slider = ui.LineDoubleSlider2D(min_value=50, max_value=50, initial_values=(50, 50))
    npt.assert_equal(slider.left_disk_value, 50)
    npt.assert_equal(slider.right_disk_value, 50)
    npt.assert_equal(slider.left_disk_ratio, 0)
    npt.assert_equal(slider.right_disk_ratio, 0)


# --- RangeSlider Tests ---


def test_range_slider_initialization():
    """Test RangeSlider creates its sub-sliders correctly."""
    slider = ui.RangeSlider()

    assert isinstance(slider.range_slider, ui.LineDoubleSlider2D)
    assert isinstance(slider.value_slider, ui.LineSlider2D)

    npt.assert_equal(slider.range_slider.left_disk_value, 0)
    npt.assert_equal(slider.range_slider.right_disk_value, 100)
    npt.assert_equal(slider.value_slider.value, 50)


def test_range_slider_actors():
    """Test that actor count matches combined sub-sliders."""
    slider = ui.RangeSlider()

    expected_count = len(slider.range_slider.actors) + len(slider.value_slider.actors)
    npt.assert_equal(len(slider.actors), expected_count)
    assert len(slider.actors) > 0


def test_range_slider_size():
    """Test that size returns positive dimensions."""
    slider = ui.RangeSlider()
    size = slider._get_size()
    assert size[0] > 0
    assert size[1] > 0


def test_range_slider_visibility():
    """Test visibility propagation to all actors."""
    slider = ui.RangeSlider()

    slider.set_visibility(False)
    for actor in slider._get_actors():
        assert actor.visible is False

    slider.set_visibility(True)
    for actor in slider._get_actors():
        assert actor.visible is True


def test_range_slider_text_precision():
    """Test that text templates match precision parameters."""
    slider = ui.RangeSlider(range_precision=2, value_precision=3)

    assert slider.range_slider_text_template == "{value:.2f}"
    assert slider.value_slider_text_template == "{value:.3f}"


def test_range_slider_scene_add():
    """Test that RangeSlider can be added to a Scene without error."""
    scene = window.Scene()
    slider = ui.RangeSlider(position=(50, 50))
    scene.add(slider)


# def test_ui_textbox(recording=False):
#     filename = "test_ui_textbox"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     print(recording_filename)
#     # TextBox
#     textbox_test = ui.TextBox2D(height=3, width=10, text="Text")

#     another_textbox_test = ui.TextBox2D(height=3, width=10, text="Enter Text")
#     another_textbox_test.set_message("Enter Text")

#     # Checking whether textbox went out of focus
#     is_off_focused = [False]

#     def _off_focus(textbox):
#         is_off_focused[0] = True

#     # Set up a callback when textbox went out of focus
#     textbox_test.off_focus = _off_focus

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(textbox_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(size=current_size, title="FURY TextBox")

#     show_manager.scene.add(textbox_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     npt.assert_equal(is_off_focused[0], True)


# def test_ui_line_slider_2d_horizontal_bottom(recording=False):
#     filename = "test_ui_line_slider_2d_horizontal_bottom"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_slider_2d_test = ui.LineSlider2D(
#         initial_value=-2,
#         min_value=-5,
#         max_value=5,
#         orientation="horizontal",
#         text_alignment="bottom",
#     )
#     line_slider_2d_test.center = (300, 300)

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(line_slider_2d_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(
#         size=current_size, title="FURY Horizontal Line Slider"
#     )

#     show_manager.scene.add(line_slider_2d_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_line_slider_2d_horizontal_top(recording=False):
#     filename = "test_ui_line_slider_2d_horizontal_top"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_slider_2d_test = ui.LineSlider2D(
#         initial_value=-2,
#         min_value=-5,
#         max_value=5,
#         orientation="horizontal",
#         text_alignment="top",
#     )
#     line_slider_2d_test.center = (300, 300)

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(line_slider_2d_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(
#         size=current_size, title="FURY Horizontal Line Slider"
#     )

#     show_manager.scene.add(line_slider_2d_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_line_slider_2d_vertical_left(recording=False):
#     filename = "test_ui_line_slider_2d_vertical_left"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_slider_2d_test = ui.LineSlider2D(
#         initial_value=-2,
#         min_value=-5,
#         max_value=5,
#         orientation="vertical",
#         text_alignment="left",
#     )
#     line_slider_2d_test.center = (300, 300)

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(line_slider_2d_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(
#         size=current_size, title="FURY Vertical Line Slider"
#     )

#     show_manager.scene.add(line_slider_2d_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_line_slider_2d_vertical_right(recording=False):
#     filename = "test_ui_line_slider_2d_vertical_right"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_slider_2d_test = ui.LineSlider2D(
#         initial_value=-2,
#         min_value=-5,
#         max_value=5,
#         orientation="vertical",
#         text_alignment="right",
#     )
#     line_slider_2d_test.center = (300, 300)

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(line_slider_2d_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(
#         size=current_size, title="FURY Vertical Line Slider"
#     )

#     show_manager.scene.add(line_slider_2d_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_2d_line_slider_hooks(recording=False):
#     global changed, value_changed, slider_moved

#     filename = "test_ui_line_slider_2d_hooks"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_slider_2d = ui.LineSlider2D(center=(300, 300))

#     event_counter = EventCounter()
#     event_counter.monitor(line_slider_2d)

#     show_manager = window.ShowManager(size=(600, 600), title="FURY Line Slider hooks")

#     # counters for the hooks to increment
#     changed = value_changed = slider_moved = 0

#     def on_line_slider_change(slider):
#         global changed
#         changed += 1

#     def on_line_slider_moved(slider):
#         global slider_moved
#         slider_moved += 1

#     def on_line_slider_value_changed(slider):
#         global value_changed
#         value_changed += 1

#     line_slider_2d.on_change = on_line_slider_change
#     line_slider_2d.on_moving_slider = on_line_slider_moved
#     line_slider_2d.on_value_changed = on_line_slider_value_changed

#     for i in range(100, -1, -1):
#         line_slider_2d.value = i

#     show_manager.scene.add(line_slider_2d)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     assert_greater(changed, 0)
#     assert_greater(value_changed, 0)
#     assert_greater(slider_moved, 0)
#     assert_equal(changed, value_changed + slider_moved)


# def test_ui_ring_slider_2d(recording=False):
#     filename = "test_ui_ring_slider_2d"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     ring_slider_2d_test = ui.RingSlider2D()
#     ring_slider_2d_test.center = (300, 300)
#     ring_slider_2d_test.value = 90

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(ring_slider_2d_test)

#     current_size = (600, 600)
#     show_manager = window.ShowManager(size=current_size, title="FURY Ring Slider")

#     show_manager.scene.add(ring_slider_2d_test)

#     if recording:
#         # Record the following events
#         # 1. Left Click on the handle and hold it
#         # 2. Move to the left the handle and make 1.5 tour
#         # 3. Release the handle
#         # 4. Left Click on the handle and hold it
#         # 5. Move to the right the handle and make 1 tour
#         # 6. Release the handle
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_2d_ring_slider_hooks(recording=False):
#     global changed, value_changed, slider_moved

#     filename = "test_ui_ring_slider_2d_hooks"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     ring_slider_2d = ui.RingSlider2D(center=(300, 300))

#     event_counter = EventCounter()
#     event_counter.monitor(ring_slider_2d)

#     show_manager = window.ShowManager(size=(600, 600), title="FURY Ring Slider hooks")

#     # counters for the ring slider changes
#     changed = value_changed = slider_moved = 0

#     def on_ring_slider_change(slider):
#         global changed
#         changed += 1

#     def on_ring_slider_moved(slider):
#         global slider_moved
#         slider_moved += 1

#     def on_ring_slider_value_changed(slider):
#         global value_changed
#         value_changed += 1

#     ring_slider_2d.on_change = on_ring_slider_change
#     ring_slider_2d.on_moving_slider = on_ring_slider_moved
#     ring_slider_2d.on_value_changed = on_ring_slider_value_changed

#     for i in range(360, -1, -1):
#         ring_slider_2d.value = i

#     show_manager.scene.add(ring_slider_2d)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     assert_greater(changed, 0)
#     assert_greater(value_changed, 0)
#     assert_greater(slider_moved, 0)
#     assert_equal(changed, value_changed + slider_moved)


# def test_ui_option(interactive=False):
#     option_test = ui.Option(label="option 1", position=(10, 10))

#     npt.assert_equal(option_test.checked, False)

#     if interactive:
#         showm = window.ShowManager(size=(600, 600))
#         showm.scene.add(option_test)
#         showm.start()


# def test_ui_checkbox_initial_state(recording=False):
#     filename = "test_ui_checkbox_initial_state"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     checkbox_test = ui.Checkbox(
#         labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
#         position=(100, 100),
#         checked_labels=["option 1", "option 4"],
#     )

#     # Collect the sequence of options that have been checked in this list.
#     selected_options = []

#     def _on_change(checkbox):
#         selected_options.append(list(checkbox.checked_labels))

#     # Set up a callback when selection changes
#     checkbox_test.on_change = _on_change

#     event_counter = EventCounter()
#     event_counter.monitor(checkbox_test)

#     # Create a show manager and record/play events.
#     show_manager = window.ShowManager(size=(600, 600), title="FURY Checkbox")
#     show_manager.scene.add(checkbox_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#         print(selected_options)
#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#         # Recorded events:
#         #  1. Click on button of option 1.
#         #  2. Click on button of option 2.
#         #  3. Click on button of option 1.
#         #  4. Click on text of option 3.
#         #  5. Click on text of option 1.
#         #  6. Click on button of option 4.
#         #  7. Click on text of option 1.
#         #  8. Click on text of option 2.
#         #  9. Click on text of option 4.
#         #  10. Click on button of option 3.
#         # Check if the right options were selected.
#         expected = [
#             ["option 4"],
#             ["option 4", "option 2\nOption 2"],
#             ["option 4", "option 2\nOption 2", "option 1"],
#             ["option 4", "option 2\nOption 2", "option 1", "option 3"],
#             ["option 4", "option 2\nOption 2", "option 3"],
#             ["option 2\nOption 2", "option 3"],
#             ["option 2\nOption 2", "option 3", "option 1"],
#             ["option 3", "option 1"],
#             ["option 3", "option 1", "option 4"],
#             ["option 1", "option 4"],
#         ]

#         npt.assert_equal(len(selected_options), len(expected))
#         assert_arrays_equal(selected_options, expected)


# def test_ui_checkbox_default(recording=False):
#     filename = "test_ui_checkbox_initial_state"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     checkbox_test = ui.Checkbox(
#         labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
#         position=(10, 10),
#         checked_labels=[],
#     )

#     old_positions = []
#     for option in checkbox_test.options.values():
#         old_positions.append(option.position)

#     old_positions = np.asarray(old_positions)
#     checkbox_test.position = (100, 100)
#     new_positions = []
#     for option in checkbox_test.options.values():
#         new_positions.append(option.position)
#     new_positions = np.asarray(new_positions)
#     npt.assert_allclose(new_positions - old_positions, 90.0 * np.ones((4, 2)))

#     # Collect the sequence of options that have been checked in this list.
#     selected_options = []

#     def _on_change(checkbox):
#         selected_options.append(list(checkbox.checked_labels))

#     # Set up a callback when selection changes
#     checkbox_test.on_change = _on_change

#     event_counter = EventCounter()
#     event_counter.monitor(checkbox_test)

#     # Create a show manager and record/play events.
#     show_manager = window.ShowManager(size=(600, 600), title="FURY Checkbox")
#     show_manager.scene.add(checkbox_test)

#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         # Recorded events:
#         #  1. Click on button of option 1.
#         #  2. Click on button of option 2.
#         #  3. Click on button of option 1.
#         #  4. Click on text of option 3.
#         #  5. Click on text of option 1.
#         #  6. Click on button of option 4.
#         #  7. Click on text of option 1.
#         #  8. Click on text of option 2.
#         #  9. Click on text of option 4.
#         #  10. Click on button of option 3.
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#         # Check if the right options were selected.
#         expected = [
#             ["option 1"],
#             ["option 1", "option 2\nOption 2"],
#             ["option 2\nOption 2"],
#             ["option 2\nOption 2", "option 3"],
#             ["option 2\nOption 2", "option 3", "option 1"],
#             ["option 2\nOption 2", "option 3", "option 1", "option 4"],
#             ["option 2\nOption 2", "option 3", "option 4"],
#             ["option 3", "option 4"],
#             ["option 3"],
#             [],
#         ]
#         npt.assert_equal(len(selected_options), len(expected))
#         assert_arrays_equal(selected_options, expected)


# def test_ui_radio_button_initial_state(recording=False):
#     filename = "test_ui_radio_button_initial"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     radio_button_test = ui.RadioButton(
#         labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
#         position=(100, 100),
#         checked_labels=["option 4"],
#     )

#     selected_option = []

#     def _on_change(radio_button):
#         selected_option.append(radio_button.checked_labels)

#     # Set up a callback when selection changes
#     radio_button_test.on_change = _on_change

#     event_counter = EventCounter()
#     event_counter.monitor(radio_button_test)

#     # Create a show manager and record/play events.
#     show_manager = window.ShowManager(size=(600, 600), title="FURY Checkbox")
#     show_manager.scene.add(radio_button_test)
#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         # Recorded events:
#         #  1. Click on button of option 1.
#         #  2. Click on button of option 2.
#         #  3. Click on button of option 2.
#         #  4. Click on text of option 2.
#         #  5. Click on button of option 1.
#         #  6. Click on text of option 3.
#         #  7. Click on button of option 4.
#         #  8. Click on text of option 4.
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#         # Check if the right options were selected.
#         expected = [
#             ["option 1"],
#             ["option 2\nOption 2"],
#             ["option 2\nOption 2"],
#             ["option 2\nOption 2"],
#             ["option 1"],
#             ["option 3"],
#             ["option 4"],
#             ["option 4"],
#         ]
#         npt.assert_equal(len(selected_option), len(expected))
#         assert_arrays_equal(selected_option, expected)


# def test_ui_radio_button_default(recording=False):
#     filename = "test_ui_radio_button_initial"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     radio_button_test = ui.RadioButton(
#         labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
#         position=(10, 10),
#         checked_labels=[],
#     )

#     old_positions = []
#     for option in radio_button_test.options.values():
#         old_positions.append(option.position)
#     old_positions = np.asarray(old_positions)
#     radio_button_test.position = (100, 100)
#     new_positions = []
#     for option in radio_button_test.options.values():
#         new_positions.append(option.position)
#     new_positions = np.asarray(new_positions)
#     npt.assert_allclose(new_positions - old_positions, 90 * np.ones((4, 2)))

#     selected_option = []

#     def _on_change(radio_button):
#         selected_option.append(radio_button.checked_labels)

#     # Set up a callback when selection changes
#     radio_button_test.on_change = _on_change

#     event_counter = EventCounter()
#     event_counter.monitor(radio_button_test)

#     # Create a show manager and record/play events.
#     show_manager = window.ShowManager(size=(600, 600), title="FURY Checkbox")
#     show_manager.scene.add(radio_button_test)
#     if recording:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         # Recorded events:
#         #  1. Click on button of option 1.
#         #  2. Click on button of option 2.
#         #  3. Click on button of option 2.
#         #  4. Click on text of option 2.
#         #  5. Click on button of option 1.
#         #  6. Click on text of option 3.
#         #  7. Click on button of option 4.
#         #  8. Click on text of option 4.
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#         # Check if the right options were selected.
#         expected = [
#             ["option 1"],
#             ["option 2\nOption 2"],
#             ["option 2\nOption 2"],
#             ["option 2\nOption 2"],
#             ["option 1"],
#             ["option 3"],
#             ["option 4"],
#             ["option 4"],
#         ]
#         npt.assert_equal(len(selected_option), len(expected))
#         assert_arrays_equal(selected_option, expected)


# def test_multiple_radio_button_pre_selected():
#     npt.assert_raises(
#         ValueError,
#         ui.RadioButton,
#         labels=["option 1", "option 2\nOption 2", "option 3", "option 4"],
#         checked_labels=["option 1", "option 4"],
#     )


# @pytest.mark.skipif(
#     True, reason="Need investigation. Incorrect number of event for each vtk version"
# )
# def test_ui_listbox_2d(interactive=False):
#     filename = "test_ui_listbox_2d"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     # Values that will be displayed by the listbox.
#     values = list(range(1, 42 + 1))
#     values.append("A Very Very Long Item To Test Text Overflow of List Box 2D")

#     if interactive:
#         listbox = ui.ListBox2D(
#             values=values,
#             size=(500, 500),
#             multiselection=True,
#             reverse_scrolling=False,
#             background_opacity=0.3,
#         )
#         listbox.center = (300, 300)
#         listbox.panel.opacity = 0.2

#         show_manager = window.ShowManager(size=(600, 600), title="FURY ListBox")
#         show_manager.scene.add(listbox)
#         show_manager.start()

#     # Recorded events:
#     #  1. Click on 1
#     #  2. Ctrl + click on 2,
#     #  3. Ctrl + click on 2.
#     #  4. Use scroll bar to scroll to the bottom.
#     #  5. Click on "A Very Very Long Item...".
#     #  6. Use scroll bar to scroll to the top.
#     #  7. Click on 1
#     #  8. Use mouse wheel to scroll down.
#     #  9. Shift + click on "A Very Very Long Item...".
#     # 10. Use mouse wheel to scroll back up.

#     listbox = ui.ListBox2D(
#         values=values, size=(500, 500), multiselection=True, reverse_scrolling=False
#     )
#     listbox.center = (300, 300)

#     # We will collect the sequence of values that have been selected.
#     selected_values = []

#     def _on_change():
#         selected_values.append(list(listbox.selected))

#     # Set up a callback when selection changes.
#     listbox.on_change = _on_change

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(listbox)

#     show_manager = window.ShowManager(size=(600, 600), title="FURY ListBox")
#     show_manager.scene.add(listbox)
#     show_manager.play_events_from_file(recording_filename)
#     expected = EventCounter.load(expected_events_counts_filename)
#     event_counter.check_counts(expected)

#     # Check if the right values were selected.
#     expected = [
#         [1],
#         [1, 2],
#         [1],
#         [
#             "A Very Very Long Item To \
# Test Text Overflow of List Box 2D"
#         ],
#         [1],
#         values,
#     ]
#     npt.assert_equal(len(selected_values), len(expected))
#     assert_arrays_equal(selected_values, expected)

#     # Test without multiselection enabled.
#     listbox.multiselection = False
#     del selected_values[:]  # Clear the list.
#     show_manager.play_events_from_file(recording_filename)

#     # Check if the right values were selected.
#     expected = [
#         [1],
#         [2],
#         [2],
#         [
#             "A Very Very Long Item To \
# Test Text Overflow of List Box 2D"
#         ],
#         [1],
#         [
#             "A Very Very Long Item To Test \
# Text Overflow of List Box 2D"
#         ],
#     ]
#     npt.assert_equal(len(selected_values), len(expected))
#     assert_arrays_equal(selected_values, expected)


# def test_ui_listbox_2d_visibility():
#     l1 = ui.ListBox2D(
#         values=["Violet", "Indigo", "Blue", "Yellow"],
#         position=(12, 10),
#         size=(100, 100),
#     )
#     l2 = ui.ListBox2D(
#         values=["Violet", "Indigo", "Blue", "Yellow"],
#         position=(10, 10),
#         size=(100, 300),
#     )

#     def assert_listbox(list_box, expected_scroll_bar_height):
#         view_end = list_box.view_offset + list_box.nb_slots
#         assert list_box.scroll_bar.height == expected_scroll_bar_height
#         for slot in list_box.slots[view_end:]:
#             assert slot.size[1] == list_box.slot_height

#     assert_listbox(l1, 40.0)

#     # Assert that for list 2 the slots and scrollbars aren't visible.
#     assert_listbox(l2, 0)


# def test_ui_file_menu_2d(interactive=False):
#     filename = "test_ui_file_menu_2d"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     with InTemporaryDirectory() as tmpdir:
#         test_dir = os.path.join(tmpdir, "testdir")
#         os.makedirs(os.path.join(test_dir, "tempdir"))
#         for i in range(10):
#             open(os.path.join(test_dir, "tempdir", f"test{i}.txt"), "w").close()
#         open(os.path.join(test_dir, "testfile.txt"), "w").close()

#         filemenu = ui.FileMenu2D(
#             size=(500, 500), extensions=["txt"], directory_path=test_dir
#         )

#         # We will collect the sequence of files that have been selected.
#         selected_files = []

#         def _on_change():
#             selected_files.append(list(filemenu.listbox.selected))

#         # Set up a callback when selection changes.
#         filemenu.listbox.on_change = _on_change

#         # Assign the counter callback to every possible event.
#         event_counter = EventCounter()
#         event_counter.monitor(filemenu)

#         # Create a show manager and record/play events.
#         show_manager = window.ShowManager(size=(600, 600), title="FURY FileMenu")
#         show_manager.scene.add(filemenu)

#         # Recorded events:
#         #  1. Click on 'testfile.txt'
#         #  2. Click on 'tempdir/'
#         #  3. Click on 'test0.txt'.
#         #  4. Shift + Click on 'test6.txt'.
#         #  5. Click on '../'.
#         #  2. Click on 'testfile.txt'.
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#         # Check if the right files were selected.
#         expected = [
#             ["testfile.txt"],
#             ["tempdir"],
#             ["test0.txt"],
#             [
#                 "test0.txt",
#                 "test1.txt",
#                 "test2.txt",
#                 "test3.txt",
#                 "test4.txt",
#                 "test5.txt",
#                 "test6.txt",
#             ],
#             ["../"],
#             ["testfile.txt"],
#         ]

#         npt.assert_equal(len(selected_files), len(expected))
#         assert_arrays_equal(selected_files, expected)
#         if interactive:
#             filemenu = ui.FileMenu2D(size=(500, 500), directory_path=os.getcwd())
#             show_manager = window.ShowManager(size=(600, 600), title="FURY FileMenu")
#             show_manager.scene.add(filemenu)
#             show_manager.start()


# def test_ui_combobox_2d(interactive=False):
#     filename = "test_ui_combobox_2d"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     values = ["An Item" + str(i) for i in range(0, 5)]
#     new_values = ["An Item5", "An Item6"]

#     combobox = ui.ComboBox2D(items=values, position=(400, 400), size=(300, 200))

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(combobox)

#     current_size = (800, 800)
#     show_manager = window.ShowManager(size=current_size, title="ComboBox UI Example")
#     show_manager.scene.add(combobox)

#     values.extend(new_values)
#     combobox.append_item(*new_values)
#     npt.assert_equal(values, combobox.items)

#     values.append("An Item7")
#     combobox.append_item("An Item7")
#     npt.assert_equal(values, combobox.items)

#     values.append("An Item8")
#     values.append("An Item9")
#     combobox.append_item("An Item8", "An Item9")
#     npt.assert_equal(values, combobox.items)

#     complex_list = [[0], (1, [[2, 3], 4], 5)]
#     combobox.append_item(*complex_list)
#     values.extend([str(i) for i in range(6)])
#     npt.assert_equal(values, combobox.items)

#     invalid_item = {"Hello": 1, "World": 2}
#     npt.assert_raises(TypeError, combobox.append_item, invalid_item)

#     npt.assert_equal(values, combobox.items)
#     npt.assert_equal((30, 20), combobox.drop_button_size)
#     npt.assert_equal([270, 140], combobox.drop_menu_size)
#     npt.assert_equal([300, 200], combobox.size)

#     ui.ComboBox2D(items=values, draggable=False)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     npt.assert_equal("An Item1", combobox.selected_text)
#     npt.assert_equal(1, combobox.selected_text_index)

#     combobox.resize((450, 300))
#     npt.assert_equal((405, 30), combobox.text_block_size)
#     npt.assert_equal((45, 30), combobox.drop_button_size)
#     npt.assert_equal((405, 210), combobox.drop_menu_size)


# def test_ui_combobox_2d_dropdown_visibility(interactive=False):
#     values = ["An Item" + str(i) for i in range(0, 5)]

#     tab_ui = ui.TabUI(position=(49, 94), size=(400, 400), nb_tabs=1, draggable=True)
#     combobox = ui.ComboBox2D(items=values, position=(400, 400), size=(300, 200))

#     tab_ui.add_element(0, combobox, (0.1, 0.3))

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(combobox)
#     event_counter.monitor(tab_ui)

#     current_size = (800, 800)
#     show_manager = window.ShowManager(size=current_size, title="ComboBox UI Example")
#     show_manager.scene.add(tab_ui)

#     tab_ui.tabs[0].content_panel.set_visibility(True)
#     npt.assert_equal(False, combobox._menu_visibility)
#     npt.assert_equal(False, combobox.drop_down_menu.panel.actors[0].GetVisibility())
#     npt.assert_equal(0, combobox.drop_down_button.current_icon_id)
#     npt.assert_equal(True, combobox.drop_down_button.actors[0].GetVisibility())
#     npt.assert_equal(True, combobox.selection_box.actors[0].GetVisibility())

#     tab_ui.tabs[0].content_panel.set_visibility(False)
#     npt.assert_equal(False, combobox._menu_visibility)
#     npt.assert_equal(False, combobox.drop_down_menu.panel.actors[0].GetVisibility())
#     npt.assert_equal(0, combobox.drop_down_button.current_icon_id)
#     npt.assert_equal(False, combobox.drop_down_button.actors[0].GetVisibility())
#     npt.assert_equal(False, combobox.selection_box.actors[0].GetVisibility())

#     iren = show_manager.scene.GetRenderWindow().GetInteractor().GetInteractorStyle()
#     combobox.menu_toggle_callback(iren, None, None)
#     tab_ui.tabs[0].content_panel.set_visibility(True)
#     npt.assert_equal(True, combobox._menu_visibility)
#     npt.assert_equal(True, combobox.drop_down_menu.panel.actors[0].GetVisibility())
#     npt.assert_equal(1, combobox.drop_down_button.current_icon_id)
#     npt.assert_equal(True, combobox.drop_down_button.actors[0].GetVisibility())
#     npt.assert_equal(True, combobox.selection_box.actors[0].GetVisibility())


# @pytest.mark.skipif(
#     skip_osx,
#     reason="This test does not work on macOS."
#     "It works on the local machines."
#     "The colors provided for shapes are "
#     "normalized values whereas when we test"
#     "it, the values returned are between "
#     "0-255. So while conversion from one"
#     "representation to another, there may be"
#     "something which causes these issues.",
# )
# def test_ui_draw_shape():
#     line = ui.DrawShape(shape_type="line", position=(150, 150))
#     quad = ui.DrawShape(shape_type="quad", position=(300, 300))
#     circle = ui.DrawShape(shape_type="circle", position=(150, 300))

#     with npt.assert_raises(IOError):
#         ui.DrawShape("poly")

#     line.resize((100, 5))
#     line.shape.color = (0, 1, 0)
#     quad.resize((150, 150))
#     quad.shape.color = (1, 0, 0)
#     circle.resize((25, 0))
#     circle.shape.color = (0, 0, 1)

#     line_color = np.round(255 * np.array(line.shape.color)).astype("uint8")
#     quad_color = np.round(255 * np.array(quad.shape.color)).astype("uint8")
#     circle_color = np.round(255 * np.array(circle.shape.color)).astype("uint8")

#     current_size = (900, 900)
#     scene = window.Scene()
#     show_manager = window.ShowManager(
#         scene=scene, size=current_size, title="DrawShape UI Example"
#     )
#     scene.add(line, circle, quad)

#     arr = window.snapshot(show_manager.scene, size=(800, 800))
#     report = window.analyze_snapshot(
#         arr, colors=[tuple(line_color), tuple(circle_color), tuple(quad_color)]
#     )
#     npt.assert_equal(report.objects, 3)
#     npt.assert_equal(report.colors_found, [True, True, True])


# def test_ui_draw_panel_basic(interactive=False):
#     filename = "test_ui_draw_panel_basic"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     drawpanel = ui.DrawPanel(size=(600, 600), position=(30, 10))

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(drawpanel)

#     current_size = (680, 680)
#     show_manager = window.ShowManager(
#         size=current_size, title="DrawPanel Basic UI Example"
#     )
#     show_manager.scene.add(drawpanel)

#     # Recorded events:
#     #  1. Check all mode selection button
#     #  2. Creation and clamping of shapes
#     #  3. Transformation and clamping of shapes

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_draw_panel_rotation(interactive=False):
#     filename = "test_ui_draw_panel_rotation"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     drawpanel = ui.DrawPanel(size=(600, 600), position=(30, 10))

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(drawpanel)

#     current_size = (680, 680)
#     show_manager = window.ShowManager(
#         size=current_size, title="DrawPanel Rotation UI Example"
#     )
#     show_manager.scene.add(drawpanel)

#     # Recorded events:
#     #  1. Rotation and clamping of shape

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_playback_panel(interactive=False):
#     global playing, paused, stopped, loop, ts

#     playing = stopped = paused = loop = False
#     ts = 0

#     current_size = (900, 620)
#     show_manager = window.ShowManager(
#         size=current_size, title="PlaybackPanel UI Example"
#     )

#     filename = "test_playback_panel"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     def play():
#         global playing
#         playing = True

#     def pause():
#         global paused
#         paused = True

#     def stop():
#         global stopped
#         stopped = True

#     def loop_toggle(value):
#         global loop
#         loop = True

#     def change_t(value):
#         global ts
#         ts = value
#         assert_greater_equal(playback.current_time, 0)
#         assert_less_equal(playback.current_time, playback.final_time)
#         assert_equal(playback.current_time, ts)

#     playback = PlaybackPanel()
#     playback.on_play = play
#     playback.on_pause = pause
#     playback.on_stop = stop
#     playback.on_loop_toggle = loop_toggle
#     playback.on_progress_bar_changed = change_t

#     show_manager.scene.add(playback)
#     event_counter = EventCounter()
#     event_counter.monitor(playback)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         event_counter.save(expected_events_counts_filename)

#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     assert_true(playing)
#     assert_true(paused)
#     assert_true(stopped)
#     assert_equal(playback.current_time, ts)
#     assert_greater(playback.current_time, 0)
#     assert_not_equal(playback.current_time_str, "00:00.00")
#     playback.current_time = 5
#     assert_equal(playback.current_time, 5)
#     assert_equal(playback.current_time_str, "00:05.00")
#     # test show/hide
#     playback.show()
#     ss = window.snapshot(show_manager.scene)
#     assert_not_equal(np.max(ss), 0)
#     playback.hide()
#     ss = window.snapshot(show_manager.scene)
#     assert_equal(np.max(ss), 0)


# def test_card_ui(interactive=False):
#     filename = "test_card_ui"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     img_url = (
#         "https://raw.githubusercontent.com/fury-gl"
#         "/fury-communication-assets/main/fury-logo.png"
#     )

#     title = "FURY"
#     body = (
#         "FURY - Free Unified Rendering in pYthon."
#         "A software library for scientific visualization in Python."
#     )

#     card = ui.elements.Card2D(
#         image_path=img_url,
#         draggable=True,
#         title_text=title,
#         body_text=body,
#         image_scale=0.5,
#     )

#     # Assign the counter callback to every possible event.

#     event_counter = EventCounter()
#     event_counter.monitor(card)

#     npt.assert_equal(card.size, (400.0, 400.0))
#     npt.assert_equal(card.image.size[1], 200.0)
#     npt.assert_equal(card.title, title)
#     npt.assert_equal(card.body, body)
#     npt.assert_equal(card.color, (0.5, 0.5, 0.5))
#     npt.assert_equal(card.panel.position, (0, 0))

#     card.title = "Changed Title"
#     npt.assert_equal(card.title, "Changed Title")

#     card.body = "Changed Body"
#     npt.assert_equal(card.body, "Changed Body")

#     card.title = title
#     card.body = body
#     card.color = (1.0, 1.0, 1.0)
#     npt.assert_equal(card.color, (1.0, 1.0, 1.0))

#     card.resize((300, 300))
#     npt.assert_equal(card.image.size[1], 150.0)
#     current_size = (600, 600)
#     show_manager = window.ShowManager(size=current_size, title="FURY Card")
#     show_manager.scene.add(card)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)


# def test_ui_spinbox(interactive=False):
#     filename = "test_ui_spinbox"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     spinbox = ui.SpinBox(size=(300, 200), min_val=-20, max_val=10, step=2)
#     npt.assert_equal(spinbox.value, 10)

#     spinbox.value = 5
#     npt.assert_equal(spinbox.value, 5)
#     spinbox.value = 50
#     npt.assert_equal(spinbox.value, 10)
#     spinbox.value = -50
#     npt.assert_equal(spinbox.value, -20)

#     spinbox.min_val = -100
#     spinbox.max_val = 100

#     spinbox.value = 5
#     npt.assert_equal(spinbox.value, 5)
#     spinbox.value = 50
#     npt.assert_equal(spinbox.value, 50)
#     spinbox.value = -50
#     npt.assert_equal(spinbox.value, -50)

#     # Assign the counter callback to every possible event.
#     event_counter = EventCounter()
#     event_counter.monitor(spinbox)

#     current_size = (800, 800)
#     show_manager = window.ShowManager(size=current_size, title="SpinBox UI Example")
#     show_manager.scene.add(spinbox)

#     if interactive:
#         show_manager.record_events_to_file(recording_filename)
#         print(list(event_counter.events_counts.items()))
#         event_counter.save(expected_events_counts_filename)
#     else:
#         show_manager.play_events_from_file(recording_filename)
#         expected = EventCounter.load(expected_events_counts_filename)
#         event_counter.check_counts(expected)

#     spinbox.resize((450, 200))
#     npt.assert_equal((315, 160), spinbox.textbox_size)
#     npt.assert_equal((90, 60), spinbox.button_size)
