"""
Visibility snapshot tests for every UI element and container.

Each element is built with real inputs (real icon assets fetched via
``fury.data``; no mocks, patches, or dummy classes), added to a real
:class:`~fury.window.Scene`, rendered offscreen, and analyzed with
:func:`fury.window.analyze_snapshot`. Visibility is toggled through each
element's real :meth:`~fury.ui.core.UI.set_visibility` API, which recurses into
child components.

The core 2D primitives (``Rectangle2D``, ``Disk2D``, ``TextBlock2D``) have their
own visibility snapshot tests in :mod:`fury.ui.tests.test_core`.
"""

import pytest

from fury import ui
from fury.actor.tests._helpers import assert_visibility
from fury.data import fetch_viz_icons, read_viz_icons
from fury.ui.core import Button2D, Slider2D


def _icon(fname="play3.png"):
    """Return a real icon image path, fetching the asset set on first use."""
    fetch_viz_icons()
    return read_viz_icons(fname=fname)


def _list_box_item():
    item = ui.ListBoxItem2D(on_select=lambda **kwargs: None, size=(100, 30))
    item.element = "Hello"
    return item


# Each entry is (id, factory). The factory builds a fresh element per call.
ELEMENT_FACTORIES = {
    "textured_button_2d": lambda: ui.TexturedButton2D(
        states={"default": _icon("play3.png")},
        position=(100, 100),
        size=(50, 50),
    ),
    "text_button_2d": lambda: ui.TextButton2D(
        label="BlueBtn", size=(120, 50), position=(50, 50)
    ),
    "line_slider_2d": lambda: ui.LineSlider2D(
        position=(100, 100), initial_value=25, min_value=0, max_value=100, length=250
    ),
    "playback_panel": lambda: ui.PlaybackPanel(position=(0, 0), width=900),
    "text_box_2d": lambda: ui.TextBox2D(
        width=10, height=3, text="Hello", position=(100, 100)
    ),
    "line_double_slider_2d": lambda: ui.LineDoubleSlider2D(
        position=(100, 100),
        initial_values=(20, 80),
        min_value=0,
        max_value=100,
        length=200,
    ),
    "ring_slider_2d": lambda: ui.RingSlider2D(
        center=(150, 150), initial_value=50, min_value=0, max_value=100
    ),
    "range_slider": lambda: ui.RangeSlider(
        range_slider_center=(450, 400),
        value_slider_center=(450, 300),
        length=200,
        min_value=0,
        max_value=100,
    ),
    "combo_box_2d": lambda: ui.ComboBox2D(
        items=["Item0", "Item1", "Item2"],
        position=(100, 200),
        size=(300, 200),
        placeholder="Pick one...",
    ),
    "list_box_2d": lambda: ui.ListBox2D(
        values=["Item 1", "Item 2", "Item 3"],
        position=(0, 0),
        size=(200, 200),
        multiselection=True,
    ),
    "list_box_item_2d": _list_box_item,
    "card_2d": lambda: ui.Card2D(
        image_path=_icon("play3.png"),
        title_text="Test Title",
        body_text="Test Body",
        size=(400, 400),
        image_scale=0.5,
        padding=10,
        border_width=2,
    ),
    "panel_2d": lambda: ui.Panel2D(size=(400, 400), position=(400, 400)),
    "tab_panel_2d": lambda: ui.TabPanel2D(size=(120, 30), title="Tab 1"),
    "tab_ui": lambda: ui.TabUI(
        position=(50, 50),
        size=(300, 300),
        tab_titles=["Tab 1", "Tab 2", "Tab 3"],
        startup_tab_id=1,
    ),
    "image_container_2d": lambda: ui.ImageContainer2D(
        img_path=_icon("home3.png"), size=(100, 100)
    ),
}


@pytest.mark.parametrize("name", list(ELEMENT_FACTORIES))
def test_element_visibility(name):
    """Every UI element renders when visible and nothing when hidden."""
    element = ELEMENT_FACTORIES[name]()
    assert_visibility(element, toggle=element.set_visibility)


def test_button2d_is_abstract():
    """Button2D is an abstract base; use TexturedButton2D / TextButton2D."""
    with pytest.raises(TypeError):
        Button2D(size=(30, 30))


def test_slider2d_is_abstract():
    """Slider2D is abstract; use LineSlider2D / RingSlider2D instead."""
    with pytest.raises(TypeError):
        Slider2D()
