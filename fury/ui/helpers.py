"""Helper variable or function for UI Elements."""

import numpy as np


TWO_PI = 2 * np.pi


def clip_overflow(textblock, width, side='right'):
    """Clips overflowing text of TextBlock2D with respect to width.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text needs to be clipped.
    width : int
        Required width of the clipped text.
    side : str, optional
        Clips the overflowing text according to side.
        It takes values "left" or "right".

    Returns
    -------
    clipped text : str
        Clipped version of the text.
    """
    side = side.lower()
    if side not in ['left', 'right']:
        raise ValueError("side can only take values 'left' or 'right'")

    original_str = textblock.message
    start_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if textblock.size[0] == width or textblock.size[0] <= width:
        textblock.have_bg = prev_bg
        return original_str

    if side == 'left':
        original_str = original_str[::-1]

    while start_ptr < end_ptr:
        mid_ptr = (start_ptr + end_ptr)//2
        textblock.message = original_str[:mid_ptr] + "..."

        is_overflowing, start_ptr, end_ptr =\
            check_overflow(textblock, width, start_ptr, end_ptr)

        if not is_overflowing:
            textblock.have_bg = prev_bg
            if side == 'left':
                textblock.message = textblock.message[::-1]
            return textblock.message


def wrap_overflow(textblock, wrap_width):
    """Wraps overflowing text of TextBlock2D with respect to width.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text needs to be wrapped.
    wrap_width : int
        Required width of the wrapped text.

    Returns
    -------
    wrapped text : str
        Wrapped version of the text.
    """
    original_str = textblock.message
    start_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if wrap_width <= 0:
        wrap_width = textblock.size[0] - abs(wrap_width) % textblock.size[0]

    if textblock.size[0] <= wrap_width:
        textblock.have_bg = prev_bg
        return original_str

    while start_ptr < end_ptr:
        mid_ptr = (start_ptr + end_ptr)//2
        textblock.message = original_str[:mid_ptr]

        is_overflowing, start_ptr, end_ptr =\
            check_overflow(textblock, wrap_width, start_ptr, end_ptr)

        if not is_overflowing:
            for i in range(len(original_str)):
                if i % mid_ptr == 0 and i != 0:
                    original_str = original_str[:i]\
                     + '\n' + original_str[i:]

            textblock.have_bg = prev_bg
            textblock.message = original_str
            return textblock.message


def check_overflow(textblock, width, start_ptr, end_ptr):
    """Checks if the text is overflowing.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text is to be checked.
    width: int
        Required width of the text.
    start_ptr : int
        Start index of the text.
    end_ptr : int
        End index of the text.

    Returns
    -------
    tuple(bool, int, int)

    """
    mid_ptr = (start_ptr + end_ptr)//2

    if textblock.size[0] < width:
        start_ptr = mid_ptr
    elif textblock.size[0] > width:
        end_ptr = mid_ptr

    if mid_ptr == (start_ptr + end_ptr)//2 or\
       textblock.size[0] == width:
        return (False, start_ptr, end_ptr)

    return (True, start_ptr, end_ptr)
