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

    clip_idx = check_overflow(textblock, width, '...', side)

    if clip_idx == 0:
        return original_str

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
    str_copy = textblock.message
    prev_bg = textblock.have_bg
    wrap_idxs = []

    wrap_idx = check_overflow(textblock, wrap_width, '', side)

    if wrap_idx == 0:
        return original_str

    wrap_idxs.append(wrap_idx)

    while wrap_idx != 0:
        str_copy = str_copy[wrap_idx:]
        textblock.message = str_copy
        wrap_idx = check_overflow(textblock, wrap_width, '', side)
        if wrap_idx != 0:
            wrap_idxs.append(wrap_idxs[-1]+wrap_idx+1)

    for idx in wrap_idxs:
        original_str = original_str[:idx] + '\n' + original_str[idx:]

    textblock.message = original_str
    textblock.have_bg = prev_bg
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
    mid_ptr: int
        Overflow index of the text.
    """
    side = side.lower()
    if side not in ['left', 'right']:
        raise ValueError("side can only take values 'left' or 'right'")

    original_str = textblock.message
    start_ptr = 0
    mid_ptr = 0
    end_ptr = len(original_str)
    prev_bg = textblock.have_bg
    textblock.have_bg = False

    if side == 'left':
        original_str = original_str[::-1]

    if textblock.size[0] <= width:
        textblock.have_bg = prev_bg
        return 0

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
            return mid_ptr
