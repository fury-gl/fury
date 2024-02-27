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
            wrap_idxs.append(wrap_idxs[-1] + wrap_idx + 1)

    for idx in wrap_idxs:
        original_str = original_str[:idx] + '\n' + original_str[idx:]

    textblock.message = original_str
    return textblock.message


def check_overflow(textblock, width, overflow_postfix='', side='right'):
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

    if side == 'left':
        original_str = original_str[::-1]

    if textblock.cal_size_from_message()[0] <= width:
        return 0

    while start_ptr < end_ptr:
        mid_ptr = (start_ptr + end_ptr) // 2
        textblock.message = original_str[:mid_ptr] + overflow_postfix

        if textblock.cal_size_from_message()[0] < width:
            start_ptr = mid_ptr
        elif textblock.cal_size_from_message()[0] > width:
            end_ptr = mid_ptr

        if (mid_ptr == (start_ptr + end_ptr) // 2 or
           textblock.cal_size_from_message()[0] == width):
            if side == 'left':
                textblock.message = textblock.message[::-1]
            return mid_ptr


def cal_bounding_box_2d(vertices):
    """Calculate the min, max position and the size of the bounding box.

    Parameters
    ----------
    vertices : ndarray
        vertices of the actors.

    """
    if vertices.ndim != 2 or vertices.shape[1] not in [2, 3]:
        raise IOError('vertices should be a 2D array with shape (n,2) or (n,3).')

    if vertices.shape[1] == 3:
        vertices = vertices[:, :-1]

    min_x, min_y = vertices[0]
    max_x, max_y = vertices[0]

    for x, y in vertices:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    bounding_box_min = np.asarray([min_x, min_y], dtype='int')
    bounding_box_max = np.asarray([max_x, max_y], dtype='int')
    bounding_box_size = np.asarray([max_x - min_x, max_y - min_y], dtype='int')

    return bounding_box_min, bounding_box_max, bounding_box_size


def rotate_2d(vertices, angle):
    """Rotate the given vertices by an angle.

    Parameters
    ----------
    vertices : ndarray
        vertices of the actors.
    angle: float
        Value by which the vertices are rotated in radian.

    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise IOError('vertices should be a 2D array with shape (n,3).')

    rotation_matrix = np.array(
        [
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    new_vertices = np.matmul(vertices, rotation_matrix)

    return new_vertices
