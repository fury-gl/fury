"""Test helpers function ."""
import numpy as np
import numpy.testing as npt

from fury import ui, window
from fury.ui.helpers import (
    cal_bounding_box_2d,
    check_overflow,
    clip_overflow,
    rotate_2d,
    wrap_overflow,
)


def test_clip_overflow():
    text = ui.TextBlock2D(text='', position=(50, 50), color=(1, 0, 0), size=(100, 50))

    sm = window.ShowManager()
    sm.scene.add(text)

    text.message = 'Hello'
    clip_overflow(text, text.size[0])
    npt.assert_equal('Hello', text.message)

    text.message = "Hello what's up?"
    clip_overflow(text, text.size[0])
    npt.assert_equal('He...', text.message)

    text.message = 'A very very long message to clip text overflow'
    clip_overflow(text, text.size[0])
    npt.assert_equal('A ...', text.message)

    text.message = 'Hello'
    clip_overflow(text, text.size[0], 'left')
    npt.assert_equal('Hello', text.message)

    text.message = 'Hello wassup'
    clip_overflow(text, text.size[0], 'left')
    npt.assert_equal('...up', text.message)

    text.message = 'A very very long message to clip text overflow'
    clip_overflow(text, text.size[0], 'left')
    npt.assert_equal('...ow', text.message)

    text.message = 'A very very long message to clip text overflow'
    clip_overflow(text, text.size[0], 'LeFT')
    npt.assert_equal('...ow', text.message)

    text.message = 'A very very long message to clip text overflow'
    clip_overflow(text, text.size[0], 'RigHT')
    npt.assert_equal('A ...', text.message)

    npt.assert_raises(ValueError, clip_overflow, text, text.size[0], 'middle')


def test_wrap_overflow():
    text = ui.TextBlock2D(text='', position=(50, 50), color=(1, 0, 0), size=(100, 50))

    sm = window.ShowManager()
    sm.scene.add(text)

    text.message = 'Hello'
    wrap_overflow(text, text.size[0])
    npt.assert_equal('Hello', text.message)

    text.message = "Hello what's up?"
    wrap_overflow(text, text.size[0])
    npt.assert_equal("Hello\n what\n's up\n?", text.message)

    text.message = 'A very very long message to clip text overflow'
    wrap_overflow(text, text.size[0])
    npt.assert_equal(
        'A ver\ny ver\ny lon\ng mes\nsage \nto cl\nip te\nxt ov\nerflo\nw', text.message
    )

    text.message = 'A very very long message to clip text overflow'
    wrap_overflow(text, 0)
    npt.assert_equal(text.message, '')

    text.message = 'A very very long message to clip text overflow'
    wrap_overflow(text, -2 * text.size[0])
    npt.assert_equal(text.message, '')


def test_check_overflow():
    text = ui.TextBlock2D(text='', position=(50, 50),
                          color=(1, 0, 0), size=(100, 50), bg_color=(.5, .5, .5))

    sm = window.ShowManager()
    sm.scene.add(text)

    text.message = 'A very very long message to clip text overflow'

    overflow_idx = check_overflow(text, 100, '~')

    npt.assert_equal(4, overflow_idx)
    npt.assert_equal('A ve~', text.message)


def test_cal_bounding_box_2d():
    vertices = np.array([[2, 2], [2, 8], [9, 3], [7, 1]])
    bb_min, bb_max, bb_size = cal_bounding_box_2d(vertices)

    npt.assert_equal([2, 1], bb_min)
    npt.assert_equal([9, 8], bb_max)
    npt.assert_equal([7, 7], bb_size)

    vertices = np.array(
        [
            [1.76, 1.11],
            [1.82, 1.81],
            [0.85, 1.94],
            [8.87, 9.57],
            [5.96, 5.51],
            [1.18, 6.79],
            [8.21, -6.67],
            [3.38, -1.06],
            [1.31, 9.61],
        ]
    )
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

    npt.assert_equal(
        np.array([[-1.0, 1.0, 0.0], [-10.0, 10.0, 0.0]], dtype='float32'),
        new_vertices.astype('float32'),
    )

    with npt.assert_raises(IOError):
        vertices = np.array([[0, 0], [0, 0]])
        new_vertices = rotate_2d(vertices, np.deg2rad(90))
