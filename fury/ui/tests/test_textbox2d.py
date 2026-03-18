import types

from fury.ui import TextBox2D, UIContext
from fury.lib import EventType


def _key_event(key: str):
    return types.SimpleNamespace(key=key, modifiers=[], type=EventType.KEY_DOWN)


def test_textbox2d_set_message_and_init_state():
    tb = TextBox2D(width=10, height=2, text="Enter Text")
    tb.set_message("abc")
    assert tb.message == "abc"
    assert tb.init is False
    assert tb.caret_pos == len("abc")


def test_textbox2d_basic_editing_and_caret_moves():
    tb = TextBox2D(width=10, height=2, text="Enter Text")
    tb.edit_mode()
    tb._key_press(_key_event("a"))
    tb._key_press(_key_event("b"))
    tb._key_press(_key_event("c"))
    assert tb.message == "abc"
    assert tb.caret_pos == 3

    tb._key_press(_key_event("ArrowLeft"))
    tb._key_press(_key_event("Backspace"))
    assert tb.message == "ac"
    assert tb.caret_pos == 1


def test_textbox2d_num_pad_inserts_digit():
    tb = TextBox2D(width=10, height=2, text="Enter Text")
    tb.edit_mode()
    tb._key_press(_key_event("Numpad3"))
    tb._key_press(_key_event("KP_1"))
    assert tb.message == "31"


def test_textbox2d_num_pad_variants_and_numlock_off():
    tb = TextBox2D(width=10, height=2, text="Enter Text")
    tb.edit_mode()
    tb._key_press(_key_event("Numpad0"))
    tb._key_press(_key_event("KP_8"))
    tb._key_press(_key_event("KP9"))
    tb._key_press(_key_event("Digit4"))
    tb._key_press(_key_event("End"))
    tb._key_press(_key_event("Down"))
    tb._key_press(_key_event("PageDown"))
    tb._key_press(_key_event("Insert"))
    assert tb.message == "08941230"


def test_textbox2d_wraps_text_for_multiline():
    tb = TextBox2D(width=3, height=2, text="Enter Text")
    tb.set_message("abcdef")
    assert tb.width_set_text("abcdef") == "abc\ndef"


def test_textbox2d_focus_and_off_focus_on_enter():
    tb = TextBox2D(width=10, height=2, text="Enter Text")
    called = {"value": False}

    def _off_focus(_ui):
        called["value"] = True

    tb.off_focus = _off_focus

    tb.edit_mode()
    assert UIContext.active_ui is tb
    tb._key_press(_key_event("Enter"))
    assert UIContext.active_ui is None
    assert called["value"] is True

