"""Test for components module."""

import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.data import fetch_viz_icons, read_viz_icons
from fury.lib import KeyboardEvent
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
    initial_tex = button.child.actor.material.map

    button.is_hovered = True
    button.update_visual_state()
    hover_tex = button.child.actor.material.map

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

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1

    mean_colors = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
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


def test_line_slider_2d_preserves_internal_z_order():
    """Test slider restores child z-order after parent z-order changes."""
    slider = ui.LineSlider2D(z_order=2)

    npt.assert_equal(slider.track.z_order, 2)
    npt.assert_equal(slider.handle.z_order, 3)
    npt.assert_equal(slider.text.z_order, 4)

    slider.z_order = 10
    npt.assert_equal(slider.track.z_order, 10)
    npt.assert_equal(slider.handle.z_order, 10)
    npt.assert_equal(slider.text.z_order, 10)

    slider.set_position((50, 50))
    npt.assert_equal(slider.track.z_order, 10)
    npt.assert_equal(slider.handle.z_order, 11)
    npt.assert_equal(slider.text.z_order, 12)


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
    """
    Test if setting visibility on the parent propagates to sub-
    components.
    """
    slider = ui.LineSlider2D()

    slider.set_visibility(False)
    for actor in slider._get_actors():
        assert actor.visible is False

    slider.set_visibility(True)
    for actor in slider._get_actors():
        assert actor.visible is True


def test_playback_panel_functional_init():
    """Test PlaybackPanel initialization and default states."""
    panel_width = 800
    playback = ui.PlaybackPanel(width=panel_width, loop=True)

    npt.assert_equal(playback._width, panel_width)
    npt.assert_equal(playback._get_size(), np.array([panel_width, 55]))

    assert playback._loop is True
    assert playback._playing is False
    npt.assert_equal(playback.speed, 1.0)
    npt.assert_equal(playback.current_time, 0)
    assert playback.current_time_str == "00:00.00"


def test_playback_panel_state_commands():
    """Test programmatic play, pause, stop, and loop commands."""
    playback = ui.PlaybackPanel()

    playback.play()
    assert playback._playing is True
    assert playback._play_pause_btn.toggled is True

    playback.pause()
    assert playback._playing is False
    assert playback._play_pause_btn.toggled is False

    playback.current_time = 50
    playback.play()
    playback.stop()
    assert playback._playing is False
    npt.assert_equal(playback.current_time, 0)

    playback.play_once()
    assert playback._loop is False
    assert playback._loop_btn.toggled is False

    playback.loop()
    assert playback._loop is True
    assert playback._loop_btn.toggled is True


def test_playback_panel_time_logic():
    """Test time formatting and slider synchronization."""
    playback = ui.PlaybackPanel()
    playback.final_time = 125  # 2 minutes, 5 seconds

    playback.current_time = 65.5
    npt.assert_equal(playback._progress_bar.value, 65.5)

    assert playback.current_time_str == "01:05.50"

    playback.final_time = 4000
    playback.current_time = 3661  # 1h, 1m, 1s
    assert playback.current_time_str == "01:01:01"


def test_playback_panel_speed_logic():
    """Test speed adjustment logic and string display."""
    playback = ui.PlaybackPanel()

    playback.speed = 2.0
    npt.assert_equal(playback.speed, 2.0)
    assert playback.speed_text.message == "2x"

    playback.speed = -1.0
    npt.assert_equal(playback.speed, 0.01)

    playback.speed = 1.500
    assert playback.speed_text.message == "1.5x"


def test_playback_panel_callback_logic():
    """Test that UI actions trigger the associated programmatic hooks."""
    playback = ui.PlaybackPanel()

    results = {"played": False, "speed": 0}

    def on_play():
        results["played"] = True

    def on_speed(val):
        results["speed"] = val

    playback.on_play = on_play
    playback.on_speed_changed = on_speed

    playback.play()
    assert results["played"] is True

    playback.speed = 4.0
    playback.on_speed_changed(playback.speed)
    npt.assert_equal(results["speed"], 4.0)


def test_playback_panel_layout_and_visibility():
    """Test visibility propagation and sub-component layout."""
    playback = ui.PlaybackPanel()

    assert playback.panel in playback._children
    assert playback._progress_bar in playback._children

    playback.set_visibility(False)
    assert playback.panel.actors[0].visible is False
    assert playback._progress_bar.track.actor.visible is False

    playback.set_visibility(True)
    assert playback.panel.actors[0].visible is True


def test_textbox_initialization():
    """Test property assignment and initial state."""
    tb = ui.TextBox2D(width=10, height=3, text="Hello")

    assert tb._message == "Hello"
    assert tb._width == 10
    assert tb._height == 3
    assert tb.init is True

    assert hasattr(tb, "text")
    assert tb.text is not None


def test_textbox_set_message():
    """Verify set_message updates internal state and actor."""
    tb = ui.TextBox2D(width=10, height=2, text="Old")
    tb.set_message("NewText")

    assert tb._message == "NewText"
    assert tb.caret_pos == len("NewText")

    assert "NewText" in tb.text.message.replace("\n", "")


def test_textbox_add_and_remove_character():
    """Test programmatic character insertion and deletion."""
    tb = ui.TextBox2D(width=5, height=1, text="")
    tb.edit_mode()

    tb.add_character("a")
    tb.add_character("b")
    tb.render_text(show_caret=False)

    assert tb._message == "ab"
    assert "ab" in tb.text.message.replace("\n", "")

    tb.remove_character()
    tb.render_text(show_caret=False)

    assert tb._message == "a"
    assert "a" in tb.text.message.replace("\n", "")


def test_textbox_caret_movement_bounds():
    """Caret should stay within valid range."""
    tb = ui.TextBox2D(width=5, height=1, text="abcd")
    tb.set_message("abcd")

    tb.move_left()
    tb.move_left()
    assert tb.caret_pos == 2

    tb.move_right()
    tb.move_right()
    assert tb.caret_pos == 4

    for _ in range(10):
        tb.move_right()
    assert tb.caret_pos == 4

    for _ in range(10):
        tb.move_left()
    assert tb.caret_pos == 0


def test_textbox_keypress_add_character():
    """Test adding characters through the handle_character method."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.handle_character("x", "x")
    tb.handle_character("y", "y")

    tb.render_text(show_caret=False)

    assert tb._message == "xy"
    assert "xy" in tb.text.message.replace("\n", "")


def test_textbox_return_triggers_off_focus():
    """Pressing Return should trigger on_blur callback."""
    tb = ui.TextBox2D(width=5, height=1, text="Start")

    called = {"off": False}

    def off_cb(event):
        called["off"] = True

    tb.on_blur = off_cb

    tb.edit_mode()
    mock_event = KeyboardEvent(type="key_down", key="Return", modifiers=[])
    tb.key_press(mock_event)

    assert called["off"] is True
    assert tb._has_focus is False


def test_textbox_render_with_caret():
    """render_text(show_caret=True) should show caret marker."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.set_message("abc")

    tb.caret_pos = 1
    tb.render_text(show_caret=True)

    assert "_" in tb.text.message


def test_textbox_wrapping_logic():
    """Test width-based wrapping logic."""
    tb = ui.TextBox2D(width=3, height=2, text="")
    tb.set_message("abcdef")

    tb.render_text(show_caret=False)

    assert tb.text.message.count("\n") >= 1


def test_textbox_visibility_propagation():
    """Visibility should propagate to internal actors."""
    tb = ui.TextBox2D(width=5, height=1, text="vis")

    tb.set_visibility(False)
    for actor in tb._get_actors():
        assert actor.visible is False

    tb.set_visibility(True)
    for actor in tb._get_actors():
        assert actor.visible is True


def test_textbox_static_background():
    """Background size should remain constant regardless of text content."""
    width, height, font_size = 10, 2, 18
    tb = ui.TextBox2D(width=width, height=height, font_size=font_size, text="Hi")

    assert tb.text.dynamic_bbox is False

    initial_bg_size = tb.text.background.size

    expected_w = int(width * font_size * 0.5)
    expected_h = int(height * font_size * 1.5) + 10
    npt.assert_equal(initial_bg_size, (expected_w, expected_h))

    tb.set_message("A" * 50)
    tb.render_text(show_caret=False)
    npt.assert_equal(tb.text.background.size, initial_bg_size)

    tb.set_message("")
    tb.render_text(show_caret=False)
    npt.assert_equal(tb.text.background.size, initial_bg_size)


def test_textbox_shift_uppercase():
    """Shift modifier should produce uppercase letters."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.handle_character("a", "a", ["Shift"])
    tb.handle_character("b", "b", ["Shift"])

    assert tb._message == "AB"


def test_textbox_capslock_uppercase():
    """CapsLock modifier should produce uppercase letters."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.handle_character("a", "a", ["CapsLock"])
    tb.handle_character("b", "b", ["CapsLock"])

    assert tb._message == "AB"


def test_textbox_shift_and_capslock_lowercase():
    """Shift + CapsLock together should produce lowercase letters."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.handle_character("a", "a", ["Shift", "CapsLock"])
    tb.handle_character("b", "b", ["Shift", "CapsLock"])

    assert tb._message == "ab"


def test_textbox_shift_symbols():
    """Shift should map number keys to their symbol equivalents."""
    tb = ui.TextBox2D(width=20, height=1, text="")
    tb.edit_mode()

    tb.handle_character("1", "1", ["Shift"])
    tb.handle_character("2", "2", ["Shift"])
    tb.handle_character("9", "9", ["Shift"])
    tb.handle_character("-", "-", ["Shift"])
    tb.handle_character(";", ";", ["Shift"])

    assert tb._message == "!@(_:"


def test_textbox_shift_enter_newline():
    """Shift+Enter should insert a newline instead of exiting."""
    tb = ui.TextBox2D(width=10, height=3, text="")
    tb.edit_mode()

    tb.handle_character("a", "a")
    result = tb.handle_character("Return", "", ["Shift"])

    assert result is False
    assert "\n" in tb._message
    assert tb._has_focus is True


def test_textbox_enter_exits_edit():
    """Enter without Shift should exit edit mode."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    called = {"off": False}
    tb.on_blur = lambda event: called.update({"off": True})

    mock_event = KeyboardEvent(type="key_down", key="Return", modifiers=[])
    tb.key_press(mock_event)

    assert tb._has_focus is False
    assert called["off"] is True


def test_textbox_width_set_text_newline_aware():
    """width_set_text should respect existing newlines."""
    tb = ui.TextBox2D(width=5, height=3, text="")

    result = tb.width_set_text("ab\ncd")
    assert result == "ab\ncd"

    result = tb.width_set_text("abcdefgh")
    assert result == "abcde\nfgh"


def test_textbox_width_set_text_empty_lines():
    """width_set_text should preserve empty lines."""
    tb = ui.TextBox2D(width=5, height=3, text="")

    result = tb.width_set_text("a\n\nb")
    assert result == "a\n\nb"


def test_textbox_render_text_overflow_clamp():
    """render_text should clamp output to self._height lines."""
    tb = ui.TextBox2D(width=5, height=2, text="")
    tb.edit_mode()

    for ch in "abcdefghijklmno":
        tb.add_character(ch)
    tb.render_text(show_caret=False)

    line_count = tb.text.message.count("\n") + 1
    assert line_count <= tb._height


def test_textbox_render_text_placeholder():
    """Empty textbox should show placeholder text."""
    tb = ui.TextBox2D(width=20, height=1, text="")
    tb.edit_mode()
    tb._message = ""
    tb.caret_pos = 0
    tb.render_text(show_caret=False)

    assert "Enter Text" in tb.text.message


def test_textbox_edit_mode_clears_init():
    """edit_mode should set init to False and enable focus."""
    tb = ui.TextBox2D(width=10, height=1, text="Hello")

    assert tb.init is True
    tb.edit_mode()
    assert tb.init is False
    assert tb._has_focus is True
    assert tb.caret_pos == len("Hello")


def test_textbox_edit_mode_clears_default_text():
    """edit_mode should clear 'Enter Text' default message."""
    tb = ui.TextBox2D(width=10, height=1, text="Enter Text")
    tb.edit_mode()

    assert tb._message == ""
    assert tb.caret_pos == 0


def test_textbox_blur():
    """blur_textbox should disable focus and trigger on_blur."""
    tb = ui.TextBox2D(width=10, height=1, text="test")

    called = {"off": False}
    tb.on_blur = lambda event: called.update({"off": True})

    tb.edit_mode()
    assert tb._has_focus is True

    tb.blur_textbox(None)
    assert tb._has_focus is False
    assert called["off"] is True


def test_textbox_blur_when_not_focused():
    """blur_textbox should be a no-op when not focused."""
    tb = ui.TextBox2D(width=10, height=1, text="test")

    called = {"off": False}
    tb.off_focus = lambda widget: called.update({"off": True})

    tb.blur_textbox()
    assert called["off"] is False


def test_textbox_move_up_down_multiline():
    """Up/down keys should move caret by one row in multiline box."""
    tb = ui.TextBox2D(width=5, height=3, text="")
    tb.edit_mode()

    for ch in "abcdefghij":
        tb.add_character(ch)

    tb.move_up()
    assert tb.caret_pos == 5

    tb.move_up()
    assert tb.caret_pos == 0

    tb.move_up()
    assert tb.caret_pos == 0

    tb.move_down()
    assert tb.caret_pos == 5


def test_textbox_move_up_down_singleline():
    """Up/down in single-line box should go to start/end."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    for ch in "hello":
        tb.add_character(ch)

    assert tb.caret_pos == 5

    tb.move_up()
    assert tb.caret_pos == 0

    tb.move_down()
    assert tb.caret_pos == 5


def test_textbox_backspace():
    """Backspace key should remove the character before the caret."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.handle_character("a", "a")
    tb.handle_character("b", "b")
    tb.handle_character("c", "c")

    tb.handle_character("Backspace", "")

    assert tb._message == "ab"


def test_textbox_backspace_at_start():
    """Backspace at position 0 should be a no-op."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()
    tb._message = "abc"
    tb.caret_pos = 0

    tb.remove_character()
    assert tb._message == "abc"
    assert tb.caret_pos == 0


def test_textbox_arrow_keys():
    """Arrow keys should move the caret via handle_character."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    for ch in "abcd":
        tb.add_character(ch)

    assert tb.caret_pos == 4

    tb.handle_character("ArrowLeft", "")
    assert tb.caret_pos == 3

    tb.handle_character("ArrowRight", "")
    assert tb.caret_pos == 4

    tb.handle_character("left", "")
    assert tb.caret_pos == 3

    tb.handle_character("right", "")
    assert tb.caret_pos == 4


def test_textbox_space_character():
    """The 'space' key should insert a space character."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.add_character("a")
    tb.add_character("space")
    tb.add_character("b")

    assert tb._message == "a b"


def test_textbox_showable_text_caret_marker():
    """showable_text with caret should include '_' marker."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.set_message("abc")
    tb.caret_pos = 1

    text_with = tb.showable_text(show_caret=True)
    text_without = tb.showable_text(show_caret=False)

    assert "_" in text_with
    assert "_" not in text_without


def test_textbox_showable_text_window_slice():
    """showable_text should only return the windowed portion."""
    tb = ui.TextBox2D(width=3, height=1, text="")
    tb.set_message("abcdef")

    text = tb.showable_text(show_caret=False)
    clean_text = text.replace("\x00", "")
    assert len(clean_text) <= tb._width * tb._height


def test_textbox_left_button_toggle():
    """Clicking should toggle between edit and blur."""
    tb = ui.TextBox2D(width=10, height=1, text="test")

    called = {"off": False}
    tb.on_blur = lambda event: called.update({"off": True})

    tb.left_button_press(None)
    assert tb._has_focus is True

    tb.left_button_press(None)
    assert tb._has_focus is False
    assert called["off"] is True


def test_textbox_multichar_key_ignored():
    """Multi-character keys (like 'Shift') should not add text."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    tb.add_character("Shift")
    assert tb._message == ""

    tb.add_character("Control")
    assert tb._message == ""


def test_textbox_handle_character_returns_false():
    """handle_character should return False for non-exit keys."""
    tb = ui.TextBox2D(width=10, height=1, text="")
    tb.edit_mode()

    result = tb.handle_character("a", "a")
    assert result is False

    result = tb.handle_character("Backspace", "")
    assert result is False


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


# def test_ui_line_double_slider_2d(interactive=False):
#     line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
#         center=(300, 300),
#         shape="disk",
#         outer_radius=15,
#         min_value=-10,
#         max_value=10,
#         initial_values=(-10, 10),
#     )
#     npt.assert_equal(line_double_slider_2d_horizontal_test.handles[0].size, (30, 30))
#     npt.assert_equal(line_double_slider_2d_horizontal_test.left_disk_value, -10)
#     npt.assert_equal(line_double_slider_2d_horizontal_test.right_disk_value, 10)

#     line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
#         center=(300, 300),
#         shape="disk",
#         outer_radius=15,
#         min_value=-10,
#         max_value=10,
#         initial_values=(-10, 10),
#     )
#     npt.assert_equal(line_double_slider_2d_vertical_test.handles[0].size, (30, 30))
#     npt.assert_equal(line_double_slider_2d_vertical_test.bottom_disk_value, -10)
#     npt.assert_equal(line_double_slider_2d_vertical_test.top_disk_value, 10)

#     if interactive:
#         show_manager = window.ShowManager(
#             size=(600, 600), title="FURY Line Double Slider"
#         )
#         show_manager.scene.add(line_double_slider_2d_horizontal_test)
#         show_manager.scene.add(line_double_slider_2d_vertical_test)
#         show_manager.start()

#     line_double_slider_2d_horizontal_test = ui.LineDoubleSlider2D(
#         center=(300, 300),
#         shape="square",
#         handle_side=5,
#         orientation="horizontal",
#         initial_values=(50, 40),
#     )
#     npt.assert_equal(line_double_slider_2d_horizontal_test.handles[0].size, (5, 5))
#     npt.assert_equal(line_double_slider_2d_horizontal_test.left_disk_value, 39)
#     npt.assert_equal(line_double_slider_2d_horizontal_test.right_disk_value, 40)
#     npt.assert_equal(line_double_slider_2d_horizontal_test.left_disk_ratio, 0.39)
#     npt.assert_equal(line_double_slider_2d_horizontal_test.right_disk_ratio, 0.4)

#     line_double_slider_2d_vertical_test = ui.LineDoubleSlider2D(
#         center=(300, 300),
#         shape="square",
#         handle_side=5,
#         orientation="vertical",
#         initial_values=(50, 40),
#     )
#     npt.assert_equal(line_double_slider_2d_vertical_test.handles[0].size, (5, 5))
#     npt.assert_equal(line_double_slider_2d_vertical_test.bottom_disk_value, 39)
#     npt.assert_equal(line_double_slider_2d_vertical_test.top_disk_value, 40)
#     npt.assert_equal(line_double_slider_2d_vertical_test.bottom_disk_ratio, 0.39)
#     npt.assert_equal(line_double_slider_2d_vertical_test.top_disk_ratio, 0.4)

#     with npt.assert_raises(ValueError):
#         ui.LineDoubleSlider2D(orientation="Not_hor_not_vert")

#     if interactive:
#         show_manager = window.ShowManager(
#             size=(600, 600), title="FURY Line Double Slider"
#         )
#         show_manager.scene.add(line_double_slider_2d_horizontal_test)
#         show_manager.scene.add(line_double_slider_2d_vertical_test)
#         show_manager.start()


# def test_ui_2d_line_double_slider_hooks(recording=False):
#     global changed, value_changed, slider_moved

#     filename = "test_ui_line_double_slider_2d_hooks"
#     recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
#     expected_events_counts_filename = pjoin(DATA_DIR, filename + ".json")

#     line_double_slider_2d = ui.LineDoubleSlider2D(center=(300, 300))

#     event_counter = EventCounter()
#     event_counter.monitor(line_double_slider_2d)

#     show_manager = window.ShowManager(
#         size=(600, 600), title="FURY Line Double Slider hooks"
#     )

#     # counters for the line double slider's changes
#     changed = value_changed = slider_moved = 0

#     def on_line_double_slider_change(slider):
#         global changed
#         changed += 1

#     def on_line_double_slider_moved(slider):
#         global slider_moved
#         slider_moved += 1

#     def on_line_double_slider_value_changed(slider):
#         global value_changed
#         value_changed += 1

#     line_double_slider_2d.on_change = on_line_double_slider_change
#     line_double_slider_2d.on_moving_slider = on_line_double_slider_moved
#     line_double_slider_2d.on_value_changed = on_line_double_slider_value_changed

#     for i in range(50, -1, -1):
#         line_double_slider_2d.left_disk_value = i
#         line_double_slider_2d.right_disk_value = 100 - i

#     show_manager.scene.add(line_double_slider_2d)

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


def test_ring_slider_2d_functional_initialization():
    """Test property assignment and initial state calculation."""
    slider = ui.RingSlider2D(
        initial_value=90,
        min_value=0,
        max_value=360,
        slider_inner_radius=40,
        slider_outer_radius=50,
        handle_outer_radius=12,
    )

    npt.assert_equal(slider.value, 90)
    npt.assert_almost_equal(slider.ratio, 90 / 360)
    npt.assert_equal(slider.min_value, 0)
    npt.assert_equal(slider.max_value, 360)

    assert isinstance(slider.handle, ui.Disk2D)
    npt.assert_equal(slider.handle.outer_radius, 12)
    npt.assert_equal(slider.track.inner_radius, 40)
    npt.assert_equal(slider.track.outer_radius, 50)


def test_ring_slider_2d_programmatic_clamping():
    """Verify that the value/ratio setters strictly enforce bounds."""
    slider = ui.RingSlider2D(min_value=0, max_value=180, initial_value=90)

    slider.value = 300
    npt.assert_equal(slider.value, 180)
    npt.assert_equal(slider.ratio, 1.0)

    slider.value = -50
    npt.assert_equal(slider.value, 0)
    npt.assert_equal(slider.ratio, 0.0)

    slider.ratio = 2.0
    npt.assert_equal(slider.ratio, 1.0)
    npt.assert_equal(slider.value, 180)

    slider.ratio = -1.0
    npt.assert_equal(slider.ratio, 0.0)
    npt.assert_equal(slider.value, 0)


def test_ring_slider_2d_synchronization():
    """Test if changing ratio updates value and vice versa."""
    slider = ui.RingSlider2D(min_value=100, max_value=200)

    slider.ratio = 0.5
    npt.assert_equal(slider.value, 150)

    slider.value = 110
    npt.assert_almost_equal(slider.ratio, 0.1)


def test_ring_slider_2d_layout_logic():
    """Verify programmatic size and handle movement logic."""
    slider = ui.RingSlider2D(
        slider_inner_radius=40,
        slider_outer_radius=60,
    )

    size = slider._get_size()

    expected_diameter = 2 * (60 + slider.handle.outer_radius)
    npt.assert_equal(size[0], expected_diameter)
    npt.assert_equal(size[1], expected_diameter)

    slider.ratio = 0.0
    pos_start = slider.handle.get_position().copy()

    slider.ratio = 0.5
    pos_mid = slider.handle.get_position().copy()

    slider.ratio = 0.75
    pos_end = slider.handle.get_position().copy()

    assert not np.allclose(pos_start, pos_mid)
    assert not np.allclose(pos_start, pos_end)


def test_ring_slider_2d_text_formatting():
    """Test the template system for programmatic text updates."""
    custom_template = "Angle: {angle:.0f}"
    slider = ui.RingSlider2D(initial_value=90, text_template=custom_template)

    assert slider.text.message == "Angle: 90"

    slider.value = 180
    assert slider.text.message == "Angle: 180"


def test_ring_slider_2d_callback_logic():
    """Test that setting values triggers the correct programmatic hooks."""
    slider = ui.RingSlider2D(initial_value=0)

    hooks_triggered = {"value_changed": False}

    def v_callback(u):
        hooks_triggered["value_changed"] = True

    slider.on_value_changed = v_callback

    slider.value = 180
    assert hooks_triggered["value_changed"] is True


def test_ui_2d_ring_slider_hooks():
    """Test that programmatic value updates trigger correct hooks."""
    slider = ui.RingSlider2D(center=(300, 300))

    changed = 0
    value_changed = 0
    slider_moved = 0

    def on_change(s):
        nonlocal changed
        changed += 1

    def on_moving(s):
        nonlocal slider_moved
        slider_moved += 1

    def on_value_changed(s):
        nonlocal value_changed
        value_changed += 1

    slider.on_change = on_change
    slider.on_moving_slider = on_moving
    slider.on_value_changed = on_value_changed

    for i in range(360, -1, -1):
        slider.value = i

    assert changed > 0
    assert value_changed > 0
    assert slider_moved == 0

    assert changed == value_changed


def test_line_double_slider_2d_initialization():
    """Test property assignment and initial state for LineDoubleSlider2D."""
    slider = ui.LineDoubleSlider2D(
        initial_values=(20, 80),
        min_value=0,
        max_value=100,
        length=200,
        line_width=5,
        outer_radius=12,
        shape="disk",
    )

    npt.assert_equal(slider.left_disk_value, 20)
    npt.assert_equal(slider.right_disk_value, 80)
    npt.assert_equal(slider.min_value, 0)
    npt.assert_equal(slider.max_value, 100)

    assert isinstance(slider.track, ui.Rectangle2D)
    assert len(slider.handles) == 2
    assert isinstance(slider.handles[0], ui.Disk2D)
    assert isinstance(slider.handles[1], ui.Disk2D)
    assert len(slider.texts) == 2


def test_line_double_slider_2d_square_handles():
    """Test LineDoubleSlider2D with square handles."""
    slider = ui.LineDoubleSlider2D(
        shape="square",
        handle_side=15,
        min_value=0,
        max_value=100,
        initial_values=(30, 70),
    )

    assert isinstance(slider.handles[0], ui.Rectangle2D)
    assert isinstance(slider.handles[1], ui.Rectangle2D)
    npt.assert_equal(slider.handles[0].size, (15, 15))
    npt.assert_equal(slider.handles[1].size, (15, 15))


def test_line_double_slider_2d_invalid_shape():
    """Test that an invalid shape raises ValueError."""
    with npt.assert_raises(ValueError):
        ui.LineDoubleSlider2D(shape="triangle")


def test_line_double_slider_2d_programmatic_value_setting():
    """Test setting left_disk_value and right_disk_value programmatically."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=100, initial_values=(0, 100))

    slider.left_disk_value = 25
    npt.assert_equal(slider.left_disk_value, 25)

    slider.right_disk_value = 75
    npt.assert_equal(slider.right_disk_value, 75)


def test_line_double_slider_2d_ratio_sync():
    """Test that ratios stay in sync with values."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=200, initial_values=(0, 200))

    slider.left_disk_value = 100
    npt.assert_almost_equal(slider._ratios[0], 0.5)

    slider.right_disk_value = 150
    npt.assert_almost_equal(slider._ratios[1], 0.75)


def test_line_double_slider_2d_initial_order_swap():
    """Test that initial_values with left > right respects the constraint."""
    slider = ui.LineDoubleSlider2D(
        min_value=0,
        max_value=100,
        initial_values=(70, 30),
    )

    npt.assert_equal(slider._values[0], 70)
    npt.assert_equal(slider._values[1], 30)


def test_line_double_slider_2d_layout_horizontal():
    """Test size calculation for horizontal orientation."""
    slider = ui.LineDoubleSlider2D(
        orientation="horizontal",
        length=300,
        line_width=5,
    )

    size = slider._get_size()
    npt.assert_equal(size[0], 300)
    assert size[1] >= 5


def test_line_double_slider_2d_layout_vertical():
    """Test size calculation for vertical orientation."""
    slider = ui.LineDoubleSlider2D(
        orientation="vertical",
        length=300,
        line_width=5,
    )

    size = slider._get_size()
    assert size[0] >= 5
    npt.assert_equal(size[1], 300)


def test_line_double_slider_2d_handle_positions_change():
    """Verify that handle positions change when values are updated."""
    slider = ui.LineDoubleSlider2D(
        min_value=0,
        max_value=100,
        initial_values=(10, 90),
        length=200,
    )

    slider.left_disk_value = 10
    left_pos_initial = slider.handles[0].get_position().copy()

    slider.left_disk_value = 50
    left_pos_moved = slider.handles[0].get_position().copy()

    assert not np.allclose(left_pos_initial, left_pos_moved)


def test_line_double_slider_2d_text_formatting():
    """Test the template system for text updates."""
    slider = ui.LineDoubleSlider2D(
        initial_values=(25, 75),
        min_value=0,
        max_value=100,
        text_template="{value:.0f}",
    )

    text_left = slider.format_text(0)
    text_right = slider.format_text(1)
    assert text_left == "25"
    assert text_right == "75"

    slider.left_disk_value = 50
    assert slider.format_text(0) == "50"


def test_line_double_slider_2d_callable_text_template():
    """Test callable text template for LineDoubleSlider2D."""

    def custom(s, idx):
        return f"H{idx}:{s._values[idx]:.0f}"

    slider = ui.LineDoubleSlider2D(
        initial_values=(10, 90),
        text_template=custom,
    )

    assert slider.format_text(0) == "H0:10"
    assert slider.format_text(1) == "H1:90"


def test_line_double_slider_2d_hooks():
    """Test that programmatic value updates trigger correct hooks."""
    slider = ui.LineDoubleSlider2D(
        min_value=0,
        max_value=100,
        initial_values=(0, 100),
    )

    on_change_count = 0
    on_value_changed_count = 0
    on_moving_count = 0

    def on_change(s):
        nonlocal on_change_count
        on_change_count += 1

    def on_value_changed(s):
        nonlocal on_value_changed_count
        on_value_changed_count += 1

    def on_moving(s):
        nonlocal on_moving_count
        on_moving_count += 1

    slider.on_change = on_change
    slider.on_value_changed = on_value_changed
    slider.on_moving_slider = on_moving

    for i in range(0, 51):
        slider.left_disk_value = i
        slider.right_disk_value = 100 - i

    assert slider.left_disk_value == 50
    assert slider.right_disk_value == 50

    assert on_moving_count == 102


def test_line_double_slider_2d_min_max_properties():
    """Test min_value and max_value getters and setters."""
    slider = ui.LineDoubleSlider2D(min_value=10, max_value=50)

    npt.assert_equal(slider.min_value, 10)
    npt.assert_equal(slider.max_value, 50)

    slider.min_value = 0
    npt.assert_equal(slider.min_value, 0)

    slider.max_value = 200
    npt.assert_equal(slider.max_value, 200)


def test_range_slider_initialization():
    """Test RangeSlider default initialization."""
    rs = ui.RangeSlider()

    assert rs.range_slider is not None
    assert rs.value_slider is not None

    assert isinstance(rs.range_slider, ui.LineDoubleSlider2D)
    assert isinstance(rs.value_slider, ui.LineSlider2D)

    npt.assert_equal(rs.min_value, 0)
    npt.assert_equal(rs.max_value, 100)


def test_range_slider_custom_params():
    """Test RangeSlider with custom parameters."""
    rs = ui.RangeSlider(
        min_value=10,
        max_value=200,
        length=400,
        shape="square",
        handle_side=15,
        orientation="horizontal",
    )

    npt.assert_equal(rs.min_value, 10)
    npt.assert_equal(rs.max_value, 200)
    npt.assert_equal(rs.length, 400)
    npt.assert_equal(rs.shape, "square")

    expected_mid = (10 + 200) / 2
    npt.assert_almost_equal(rs.value_slider.value, expected_mid)


def test_range_slider_vertical():
    """Test RangeSlider with vertical orientation."""
    rs = ui.RangeSlider(orientation="vertical")

    npt.assert_equal(rs.orientation, "vertical")


def test_range_slider_subcomponent_structure():
    """Test that the RangeSlider composes a LineDoubleSlider2D and LineSlider2D."""
    rs = ui.RangeSlider(shape="disk")

    assert len(rs.range_slider.handles) == 2

    assert rs.value_slider.handle is not None


def test_range_slider_range_slider_callback():
    """Test that setting range slider values auto-updates value slider bounds."""
    rs = ui.RangeSlider(min_value=0, max_value=100)

    rs.range_slider.left_disk_value = 20
    rs.range_slider.right_disk_value = 80

    npt.assert_equal(rs.value_slider.min_value, 20)
    npt.assert_equal(rs.value_slider.max_value, 80)


def test_range_slider_value_clamping_after_range_change():
    """Test that the value slider clamps its value after range change."""
    rs = ui.RangeSlider(min_value=0, max_value=100)

    rs.value_slider.value = 90

    rs.range_slider.left_disk_value = 20
    rs.range_slider.right_disk_value = 60

    assert rs.value_slider.value <= 60


def test_range_slider_size():
    """Test RangeSlider size computation."""
    rs = ui.RangeSlider(length=200)

    size = rs._get_size()
    assert size[0] > 0
    assert size[1] > 0


def test_range_slider_text_templates():
    """Test that custom precision templates are applied."""
    rs = ui.RangeSlider(
        range_precision=2,
        value_precision=3,
    )

    assert "{value:.2f}" == rs.range_slider_text_template
    assert "{value:.3f}" == rs.value_slider_text_template


def test_line_double_slider_2d_min_max_validation():
    """Test that min/max setters raise ValueError on invalid values."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=100)

    with npt.assert_raises(ValueError):
        slider.min_value = 100

    with npt.assert_raises(ValueError):
        slider.min_value = 150

    with npt.assert_raises(ValueError):
        slider.max_value = 0

    with npt.assert_raises(ValueError):
        slider.max_value = -10


def test_line_double_slider_2d_value_clamping():
    """Test that disk value setters clamp to [min_value, max_value]."""
    slider = ui.LineDoubleSlider2D(min_value=0, max_value=100, initial_values=(0, 100))

    slider.left_disk_value = -50
    npt.assert_equal(slider.left_disk_value, 0)

    slider.left_disk_value = 200
    npt.assert_equal(slider.left_disk_value, 100)

    slider.right_disk_value = -50
    npt.assert_equal(slider.right_disk_value, 0)

    slider.right_disk_value = 200
    npt.assert_equal(slider.right_disk_value, 100)


def test_line_double_slider_2d_initial_values_clamping():
    """Test that out-of-range initial_values are clamped."""
    slider = ui.LineDoubleSlider2D(
        min_value=10, max_value=90, initial_values=(-50, 200)
    )

    npt.assert_equal(slider.left_disk_value, 10)
    npt.assert_equal(slider.right_disk_value, 90)
    npt.assert_almost_equal(slider._ratios[0], 0.0)
    npt.assert_almost_equal(slider._ratios[1], 1.0)


def test_range_slider_size_horizontal():
    """Test that horizontal RangeSlider adds heights of sub-sliders."""
    rs = ui.RangeSlider(length=200, orientation="horizontal")

    size = rs._get_size()
    expected_h = rs.range_slider.size[1] + rs.value_slider.size[1]
    expected_w = max(rs.range_slider.size[0], rs.value_slider.size[0])
    npt.assert_equal(size[0], expected_w)
    npt.assert_equal(size[1], expected_h)


def test_range_slider_size_vertical():
    """Test that vertical RangeSlider adds widths of sub-sliders."""
    rs = ui.RangeSlider(length=200, orientation="vertical")

    size = rs._get_size()
    expected_w = rs.range_slider.size[0] + rs.value_slider.size[0]
    expected_h = max(rs.range_slider.size[1], rs.value_slider.size[1])
    npt.assert_equal(size[0], expected_w)
    npt.assert_equal(size[1], expected_h)


# def test_ui_range_slider(interactive=False):
#     range_slider_test_horizontal = ui.RangeSlider(shape="square")
#     range_slider_test_vertical = ui.RangeSlider(shape="square", orientation="vertical")  # noqa: E501

#     if interactive:
#         show_manager = window.ShowManager(
#             size=(600, 600), title="FURY Line Double Slider"
#         )
#         show_manager.scene.add(range_slider_test_horizontal)
#         show_manager.scene.add(range_slider_test_vertical)
#         show_manager.start()


# def test_ui_slider_value_range():
#     with npt.assert_no_warnings():
#         # LineSlider2D
#         line_slider = ui.LineSlider2D(min_value=0, max_value=0)
#         assert_equal(line_slider.value, 0)
#         assert_equal(line_slider.min_value, 0)
#         assert_equal(line_slider.max_value, 0)
#         line_slider.value = 100
#         assert_equal(line_slider.value, 0)
#         line_slider.value = -100
#         assert_equal(line_slider.value, 0)

#         line_slider = ui.LineSlider2D(min_value=0, max_value=100)
#         line_slider.value = 105
#         assert_equal(line_slider.value, 100)
#         line_slider.value = -100
#         assert_equal(line_slider.value, 0)

#         # LineDoubleSlider2D
#         line_double_slider = ui.LineDoubleSlider2D(min_value=0, max_value=0)
#         assert_equal(line_double_slider.left_disk_value, 0)
#         assert_equal(line_double_slider.right_disk_value, 0)
#         line_double_slider.left_disk_value = 100
#         assert_equal(line_double_slider.left_disk_value, 0)
#         line_double_slider.right_disk_value = -100
#         assert_equal(line_double_slider.right_disk_value, 0)

#         line_double_slider = ui.LineDoubleSlider2D(min_value=50, max_value=100)
#         line_double_slider.right_disk_value = 150
#         assert_equal(line_double_slider.right_disk_value, 100)
#         line_double_slider.left_disk_value = -150
#         assert_equal(line_double_slider.left_disk_value, 50)

#         # RingSlider2D
#         ring_slider = ui.RingSlider2D(initial_value=0, min_value=0, max_value=0)
#         assert_equal(ring_slider.value, 0)
#         assert_equal(ring_slider.previous_value, 0)
#         ring_slider.value = 180
#         assert_equal(ring_slider.value, 0)
#         ring_slider.value = -180
#         assert_equal(ring_slider.value, 0)

#         # RangeSlider
#         range_slider_2d = ui.RangeSlider(min_value=0, max_value=0)
#         assert_equal(range_slider_2d.value_slider.value, 0)
#         range_slider_2d.value_slider.value = 100
#         assert_equal(range_slider_2d.value_slider.value, 0)


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


def test_ui_combobox_2d_initialization_and_properties():
    """Test ComboBox2D initialization, default parameters, layout, and properties."""
    fetch_viz_icons()
    values = ["Item0", "Item1", "Item2"]

    combobox = ui.ComboBox2D(
        items=values, position=(100, 100), size=(300, 200), placeholder="Pick one..."
    )
    npt.assert_equal(combobox.items, values)
    npt.assert_equal(combobox.selected_text, "Pick one...")
    npt.assert_equal(combobox._menu_visibility, False)
    assert combobox.selected_text_index is None
    assert combobox._drag_offset is None
    npt.assert_equal(combobox.size, (300, 200))
    npt.assert_equal(combobox.panel_size, (300, 200))
    assert combobox.panel in combobox._children

    default_cb = ui.ComboBox2D(items=["placeholder"])
    npt.assert_equal(default_cb._selection, "Choose selection...")
    npt.assert_equal(default_cb.draggable, True)
    npt.assert_equal(default_cb.font_size, 20)

    combobox.resize((450, 300))
    expected_text = (int(0.85 * 450), int(0.2 * 300))
    expected_menu = (450, int(0.8 * 300))
    expected_btn = (int(0.15 * 450), int(0.2 * 300))

    npt.assert_equal(combobox.text_block_size, expected_text)
    npt.assert_equal(combobox.drop_menu_size, expected_menu)
    npt.assert_equal(combobox.drop_button_size, expected_btn)

    # Test font scaling
    expected_font_size = int(20 * (300 / 200))  # 30
    npt.assert_equal(combobox.font_size, expected_font_size)
    npt.assert_equal(combobox.drop_down_menu.font_size, expected_font_size)
    npt.assert_equal(combobox.selection_box.font_size, expected_font_size)
    expected_slot_height = max(
        1, int(expected_font_size * combobox.drop_down_menu.line_spacing)
    )
    npt.assert_equal(combobox.drop_down_menu.slot_height, expected_slot_height)

    # Test appending
    combobox.append_item("C", "D")
    npt.assert_equal(combobox.items, ["Item0", "Item1", "Item2", "C", "D"])
    complex_list = [[1], (2, [[3, 4], 5])]
    combobox.append_item(*complex_list)
    combobox.append_item(42, 3.14)
    expected = [
        "Item0",
        "Item1",
        "Item2",
        "C",
        "D",
        "1",
        "2",
        "3",
        "4",
        "5",
        "42",
        "3.14",
    ]
    npt.assert_equal(combobox.items, expected)
    npt.assert_equal(combobox.drop_down_menu.values, expected)

    npt.assert_raises(TypeError, combobox.append_item, {"key": "value"})

    # Test items list copy
    original = ["A", "B"]
    copy_cb = ui.ComboBox2D(items=original)
    original.append("C")
    npt.assert_equal(copy_cb.items, ["A", "B"])


def test_ui_combobox_2d_interaction():
    """Test menu toggle, selection, and callbacks."""
    fetch_viz_icons()
    combobox = ui.ComboBox2D(items=["A", "B", "C"])

    # Test Toggle
    mock_event = window.PointerEvent(
        x=100, y=100, type=window.EventType.POINTER_DOWN, target="target"
    )
    combobox.menu_toggle_callback(mock_event)
    assert combobox._menu_visibility is True
    assert combobox.drop_down_button.toggled is True

    combobox.menu_toggle_callback(mock_event)
    assert combobox._menu_visibility is False

    # Test selection
    results = {"called": False, "selected": None}

    def on_change(cb):
        results["called"] = True
        results["selected"] = cb.selected_text

    combobox.on_change = on_change

    combobox.drop_down_menu.selected = ["B"]
    combobox.drop_down_menu.last_selection_idx = 1
    combobox._menu_visibility = True
    combobox.select_option_callback()

    npt.assert_equal(combobox.selected_text, "B")
    npt.assert_equal(combobox.selected_text_index, 1)
    assert combobox._menu_visibility is False
    assert results["called"] is True
    npt.assert_equal(results["selected"], "B")

    combobox.drop_down_menu.selected = []
    combobox.select_option_callback()
    npt.assert_equal(combobox.selected_text, "B")


def test_ui_combobox_2d_visibility():
    """Test set_visibility logic."""
    fetch_viz_icons()
    combobox = ui.ComboBox2D(items=["A", "B", "C"])

    combobox.set_visibility(False)
    assert combobox.panel.actors[0].visible is False
    combobox.set_visibility(True)
    assert combobox.panel.actors[0].visible is True
    assert combobox._menu_visibility is False

    combobox._menu_visibility = True
    combobox.drop_down_menu.set_visibility(True)
    combobox._menu_visibility = False
    combobox.set_visibility(True)
    assert combobox._menu_visibility is False


def test_ui_combobox_2d_drag_events():
    """Test drag properties and operations."""
    fetch_viz_icons()
    combobox = ui.ComboBox2D(items=["A"], position=(10, 10), draggable=True)

    event_press = window.PointerEvent(
        x=20, y=20, type=window.EventType.POINTER_DOWN, target="target"
    )
    combobox.left_button_pressed(event_press)
    npt.assert_array_almost_equal(combobox._drag_offset, [10, 10])

    event_drag = window.PointerEvent(
        x=50, y=60, type=window.EventType.POINTER_MOVE, target="target"
    )
    combobox.left_button_dragged(event_drag)
    npt.assert_array_almost_equal(combobox.get_position(), [40, 50])

    cb2 = ui.ComboBox2D(items=["A"], position=(10, 10), draggable=True)
    cb2.left_button_dragged(event_drag)
    npt.assert_array_almost_equal(cb2.get_position(), [10, 10])

    cb3 = ui.ComboBox2D(items=["A"], draggable=False)
    assert cb3.draggable is False


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


def test_listbox_2d_functional_initialization():
    values = [
        "Item 1",
        "Item 2",
        "A extremely long string that will trigger the clip overflow algorithm",
    ]
    listbox = ui.ListBox2D(
        values=values, position=(0, 0), size=(200, 200), multiselection=True
    )

    npt.assert_equal(len(listbox.values), 3)
    npt.assert_equal(listbox.size, [200, 200])
    npt.assert_equal(len(listbox.slots), listbox.nb_slots)

    npt.assert_equal(listbox.slots[0].element, "Item 1")
    npt.assert_equal(listbox.slots[1].element, "Item 2")
    npt.assert_equal(
        listbox.slots[2].element,
        "A extremely long string that will trigger the clip overflow algorithm",
    )


def test_listbox_2d_visibility_and_scrollbar():
    values_few = ["Item 1", "Item 2"]
    values_many = [f"Item {i}" for i in range(50)]

    listbox_few = ui.ListBox2D(values=values_few, size=(100, 300))
    npt.assert_equal(listbox_few.scroll_bar.height, 1)
    npt.assert_equal(listbox_few.scroll_bar.actors[0].visible, False)

    listbox_many = ui.ListBox2D(values=values_many, size=(100, 300))
    npt.assert_equal(listbox_many.scroll_bar.actors[0].visible, True)
    assert listbox_many.scroll_bar.height > 0


def test_listbox_2d_selection_logic():
    values = ["Item 1", "Item 2", "Item 3"]
    listbox = ui.ListBox2D(values=values, size=(100, 300), multiselection=True)

    callbacks_triggered = []

    def on_change():
        callbacks_triggered.append(list(listbox.selected))

    listbox.on_change = on_change

    listbox.select(listbox.slots[0])
    npt.assert_equal(listbox.slots[0].selected, True)
    npt.assert_equal(listbox.selected, ["Item 1"])
    npt.assert_equal(callbacks_triggered[-1], ["Item 1"])

    listbox.select(listbox.slots[1], multiselect=True)
    npt.assert_equal(listbox.slots[1].selected, True)
    npt.assert_equal(listbox.selected, ["Item 1", "Item 2"])

    listbox.select(listbox.slots[2], multiselect=False)
    npt.assert_equal(listbox.slots[0].selected, False)
    npt.assert_equal(listbox.slots[1].selected, False)
    npt.assert_equal(listbox.slots[2].selected, True)
    npt.assert_equal(listbox.selected, ["Item 3"])


def test_listbox_2d_scrolling_logic():
    values = [f"Item {i}" for i in range(50)]
    listbox = ui.ListBox2D(values=values, size=(100, 300))

    npt.assert_equal(listbox.view_offset, 0)

    for _ in range(5):
        listbox.scroll_down()

    npt.assert_equal(listbox.view_offset, 5)
    npt.assert_equal(listbox.slots[0].element, "Item 5")

    for _ in range(2):
        listbox.scroll_up()

    npt.assert_equal(listbox.view_offset, 3)

    for _ in range(10):
        listbox.scroll_up()

    npt.assert_equal(listbox.view_offset, 0)


def test_listbox_2d_resize():
    values = [f"Item {i}" for i in range(10)]
    listbox = ui.ListBox2D(values=values, size=(100, 200))

    npt.assert_equal(listbox.size, [100, 200])

    new_size = (200, 400)
    listbox.resize(new_size)

    npt.assert_equal(listbox.size, [200, 400])
    npt.assert_equal(listbox.panel.size, new_size)

    new_scrollbar_width = int(new_size[0] / 20)
    expected_scroll_bar_x = int(new_size[0] - new_scrollbar_width - listbox.margin)
    npt.assert_equal(listbox._scroll_bar_x, expected_scroll_bar_x)

    expected_slot_width = int(new_size[0] - new_scrollbar_width - 3 * listbox.margin)
    npt.assert_equal(listbox.slot_width, expected_slot_width)

    npt.assert_equal(len(listbox.slots), listbox.nb_slots)


def test_ui_card2d_initialization():
    """Test Card2D initialization and layout logic."""
    fetch_viz_icons()
    img_path = read_viz_icons(fname="play3.png")

    card = ui.Card2D(
        image_path=img_path,
        title_text="Test Title",
        body_text="Test Body",
        size=(400, 400),
        image_scale=0.5,
        padding=10,
        border_width=2,
    )

    npt.assert_equal(card.card_size, (400, 400))
    npt.assert_equal(card.title, "Test Title")
    npt.assert_equal(card.body, "Test Body")

    assert isinstance(card.image, ui.ImageContainer2D)
    assert isinstance(card.title_box, ui.TextBlock2D)
    assert isinstance(card.body_box, ui.TextBlock2D)
    assert isinstance(card.panel, ui.Panel2D)

    img_w = max(400 - 2 * 2, 1)
    img_h = max(int(0.5 * 400), 1)

    npt.assert_equal(card._image_size, (img_w, img_h))
    npt.assert_equal(card.image.size, (img_w, img_h))


def test_ui_card2d_properties():
    """Test Card2D getters and setters."""
    fetch_viz_icons()
    img_path = read_viz_icons(fname="play3.png")

    card = ui.Card2D(
        image_path=img_path,
        title_text="Old Title",
        body_text="Old Body",
        bg_color=(1.0, 0.0, 0.0),
        bg_opacity=0.5,
    )

    card.title = "New Title"
    npt.assert_equal(card.title, "New Title")
    npt.assert_equal(card.title_box.message, "New Title")

    card.body = "New Body"
    npt.assert_equal(card.body, "New Body")
    npt.assert_equal(card.body_box.message, "New Body")

    card.color = (0.0, 1.0, 0.0)
    npt.assert_almost_equal(card.color, (0.0, 1.0, 0.0))

    card.opacity = 0.8
    npt.assert_almost_equal(card.opacity, 0.8)


def test_ui_card2d_resize():
    """Test Card2D resize propagates to internal components."""
    fetch_viz_icons()
    img_path = read_viz_icons(fname="play3.png")

    card = ui.Card2D(image_path=img_path, size=(200, 200), border_width=0, padding=0)

    new_size = (300, 400)
    card.resize(new_size)

    npt.assert_equal(card.card_size, new_size)
    npt.assert_equal(card.panel.size, new_size)

    npt.assert_equal(card.image.size, (300, 200))
    npt.assert_equal(card._image_size, (300, 200))


def test_ui_card2d_events():
    """Test Card2D dragging events."""
    fetch_viz_icons()
    img_path = read_viz_icons(fname="play3.png")

    card = ui.Card2D(image_path=img_path, position=(10, 10))

    event_press = window.PointerEvent(
        x=20, y=20, type=window.EventType.POINTER_DOWN, target="target"
    )
    card.left_button_pressed(event_press)

    npt.assert_array_almost_equal(card._drag_offset, [10, 10])

    event_drag = window.PointerEvent(
        x=50, y=60, type=window.EventType.POINTER_MOVE, target="target"
    )
    card.left_button_dragged(event_drag)

    npt.assert_array_almost_equal(card.get_position(), [40, 50])
