"""Test helpers fonction ."""

import numpy.testing as npt

from fury import window, ui
from fury.ui.helpers import clip_overflow


def test_clip_overflow():
    text = ui.TextBlock2D(text="", position=(50, 50), color=(1, 0, 0))
    rectangle = ui.Rectangle2D(position=(50, 50), size=(100, 50))

    sm = window.ShowManager()
    sm.scene.add(rectangle, text)

    text.message = "Hello"
    clip_overflow(text, rectangle.size[0])
    npt.assert_equal("Hello", text.message)

    text.message = "Hello wassup"
    clip_overflow(text, rectangle.size[0])
    npt.assert_equal("Hello was...", text.message)

    text.message = "A very very long message to clip text overflow"
    clip_overflow(text, rectangle.size[0])
    npt.assert_equal("A very ve...", text.message)

    text.message = "Hello"
    clip_overflow(text, rectangle.size[0], 'left')
    npt.assert_equal("Hello", text.message)

    text.message = "Hello wassup"
    clip_overflow(text, rectangle.size[0], 'left')
    npt.assert_equal("...lo wassup", text.message)

    text.message = "A very very long message to clip text overflow"
    clip_overflow(text, rectangle.size[0], 'left')
    npt.assert_equal("... overflow", text.message)

    text.message = "A very very long message to clip text overflow"
    clip_overflow(text, rectangle.size[0], 'LeFT')
    npt.assert_equal("... overflow", text.message)

    text.message = "A very very long message to clip text overflow"
    clip_overflow(text, rectangle.size[0], 'RigHT')
    npt.assert_equal("A very ve...", text.message)

    npt.assert_raises(ValueError, clip_overflow,
                      text, rectangle.size[0], 'middle')
