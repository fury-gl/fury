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

from fury.testing import assert_visibility
from fury.ui.core import Button2D, Slider2D
from fury.ui.tests._helpers import ELEMENT_FACTORIES


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
