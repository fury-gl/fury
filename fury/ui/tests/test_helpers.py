"""Test helpers fonction ."""

from fury.ui.core import UI
import numpy.testing as npt

from fury import window, ui
from fury.ui.helpers import clip_overflow, is_ui


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


class DummyActor:
    def __init__(self, act):
        self._act = act

    @property
    def act(self):
        return self._act

    def add_to_scene(self, ren):
        """ Adds the items of this container to a given scene. """
        return ren


class DummyUI(UI):
    def __init__(self, act):
        super(DummyUI, self).__init__()
        self.act = act

    def _setup(self):
        pass

    def _get_actors(self):
        return []

    def _add_to_scene(self, scene):
        return scene

    def _get_size(self):
        return (5, 5)

    def _set_position(self, coords):
        return coords


def test_is_ui():
    panel = ui.Panel2D(position=(0, 0), size=(100, 100))
    grid = DummyUI(act=[])
    container = DummyActor(act="act")

    npt.assert_equal(True, is_ui(panel))
    npt.assert_equal(False, is_ui(container))
    npt.assert_equal(True, is_ui(grid))
