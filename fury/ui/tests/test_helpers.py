"""Test helpers fonction ."""
import numpy as np
import numpy.testing as npt

from fury import window, ui
from fury.ui.helpers import (clip_overflow, wrap_overflow, check_overflow,
                             cal_bounding_box_2d, rotate_2d)


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


def test_wrap_overflow():
    text = ui.TextBlock2D(text="", position=(50, 50), color=(1, 0, 0))
    rectangle = ui.Rectangle2D(position=(50, 50), size=(100, 50))

    sm = window.ShowManager()
    sm.scene.add(rectangle, text)

    text.message = "Hello"
    wrap_overflow(text, rectangle.size[0])
    npt.assert_equal("Hello", text.message)

    text.message = "Hello wassup"
    wrap_overflow(text, rectangle.size[0])
    npt.assert_equal("Hello wassu\np", text.message)

    text.message = "A very very long message to clip text overflow"
    wrap_overflow(text, rectangle.size[0])
    npt.assert_equal("A very very\n long mess\nage to clip \ntext overflo\nw",
                     text.message)

    text.message = "A very very long message to clip text overflow"
    wrap_overflow(text, 0)
    npt.assert_equal(text.message, "")

    wrap_overflow(text, -2*text.size[0])
    npt.assert_equal(text.message, "")


def test_check_overflow():
    text = ui.TextBlock2D(text="", position=(50, 50), color=(1, 0, 0))
    rectangle = ui.Rectangle2D(position=(50, 50), size=(100, 50))

    sm = window.ShowManager()
    sm.scene.add(rectangle, text)

    text.message = "A very very long message to clip text overflow"

    overflow_idx = check_overflow(text, rectangle.size[0], '~')

    npt.assert_equal(10, overflow_idx)
    npt.assert_equal('A very ver~', text.message)


def test_cal_bounding_box_2d():
    vertices = np.array([[2, 2], [2, 8], [9, 3], [7, 1]])
    bb_min, bb_max, bb_size = cal_bounding_box_2d(vertices)

    npt.assert_equal([2, 1], bb_min)
    npt.assert_equal([9, 8], bb_max)
    npt.assert_equal([7, 7], bb_size)

    vertices = np.array([[1.76, 1.11], [1.82, 1.81], [0.85, 1.94],
                         [8.87, 9.57], [5.96, 5.51], [1.18, 6.79],
                         [8.21, -6.67], [3.38, -1.06], [1.31, 9.61]])
    bb_min, bb_max, bb_size = cal_bounding_box_2d(vertices)

    npt.assert_equal([0, -6], bb_min)
    npt.assert_equal([8, 9], bb_max)
    npt.assert_equal([8, 16], bb_size)

    with npt.assert_raises(IOError):
        vertices = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        bb_min, bb_max, bb_size = cal_bounding_box_2d(vertices)


def test_rotate_2d():
    vertices = np.array([[1, 1, 0], [10, 10, 0]])
    new_vertices = rotate_2d(vertices, np.deg2rad(90))

    npt.assert_equal(np.array([[-1., 1., 0.], [-10., 10., 0.]],
                     dtype="float32"), new_vertices.astype("float32"))

    with npt.assert_raises(IOError):
        vertices = np.array([[0, 0], [0, 0]])
        new_vertices = rotate_2d(vertices, np.deg2rad(90))
