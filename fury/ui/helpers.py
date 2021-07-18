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
    original_str = textblock.message
    prev_bg = textblock.have_bg

    is_overflowing, mid_ptr = check_overflow(textblock, width, '...', side)

    if mid_ptr == 0:
        return original_str

    if not is_overflowing:
        textblock.have_bg = prev_bg
        return textblock.message


def wrap_overflow(textblock, wrap_width, side='right'):
    """Wraps overflowing text of TextBlock2D with respect to width.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text needs to be wrapped.
    wrap_width : int
        Required width of the wrapped text.
    side : str, optional
        Clips the overflowing text according to side.
        It takes values "left" or "right".

    Returns
    -------
    wrapped text : str
        Wrapped version of the text.
    """
    original_str = textblock.message
    prev_bg = textblock.have_bg

    is_overflowing, mid_ptr = check_overflow(textblock, wrap_width, '', side)

    if mid_ptr == 0:
        return original_str

    if not is_overflowing:
        for i in range(len(original_str)):
            if i % mid_ptr == 0 and i != 0:
                original_str = original_str[:i]\
                    + '\n' + original_str[i:]

        textblock.have_bg = prev_bg
        textblock.message = original_str
        return textblock.message


def check_overflow(textblock, width, overflow_postfix='',
                   side='right'):
    """Checks if the text is overflowing.

    Parameters
    ----------
    textblock : TextBlock2D
        The textblock object whose text is to be checked.
    width: int
        Required width of the text.
    overflow_postfix: str, optional
        Postfix to be added to the text if it is overflowing.

    Returns
    -------
    tuple(bool, int)

    """
    side = side.lower()
    if side not in ['left', 'right']:
        raise ValueError("side can only take values 'left' or 'right'")

    original_str = textblock.message
    start_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if side == 'left':
        original_str = original_str[::-1]

    if textblock.size[0] <= width:
        textblock.have_bg = prev_bg
        return (False, 0)

    while start_ptr < end_ptr:
        mid_ptr = (start_ptr + end_ptr)//2
        textblock.message = original_str[:mid_ptr] + overflow_postfix

        if textblock.size[0] < width:
            start_ptr = mid_ptr
        elif textblock.size[0] > width:
            end_ptr = mid_ptr

        if mid_ptr == (start_ptr + end_ptr) // 2 or textblock.size[0] == width:
            if side == 'left':
                textblock.message = textblock.message[::-1]
            return (False, mid_ptr)
