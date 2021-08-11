import fury
import numpy as np
from PIL import ImageFont, ImageDraw, Image


num_ascii_chars = 95
num_cols_ascii = 9
num_rows_ascii = 11


def get_ascii_chars():
    """This function is used to generate a list of ascii characters.

    Returns
    -------
    chars : ndarray
        A numpy array of ascii characters.
    char2pos : dict
        A dictionary that maps characters to their positions in the
        numpy array.

    """
    ascii_chars = [
        chr(i) for i in range(32, 127)
    ]

    chars = np.zeros(shape=(num_rows_ascii, num_cols_ascii), dtype='str')
    char2pos = {}
    for i in range(num_rows_ascii):
        for j in range(num_cols_ascii):
            index = i*num_cols_ascii+j
            if index >= num_ascii_chars-1:
                chars[i, j] = ' '
                continue
            char = ascii_chars[index]
            chars[i, j] = char
            char2pos[char] = [i/num_rows_ascii, j/num_cols_ascii][::-1]
    return [
        chars,
        char2pos,
    ]


def create_bitmap_font(
        font_size=50, font_path=None, background_color='black', pad=0,
        show=False, save_path=None):
    """This function is used to create a bitmap font.

    Parameters
    ----------
    font_size : int
        The size of the font.
    font_path : str
        The path to the font file.
    background_color : str
        The color of the background.
    pad : int
        The padding of the font.
    show : bool
        Whether to show the result.
    save_path : str
        The path to save the image file.

    Returns
    -------
    image_array : ndarray
        The image array.
    char2pos : dict
        A dictionary that maps characters to their positions in the
        numpy array.

    """

    chars, char2pos = get_ascii_chars()
    if font_size == 50 and font_path is None:
        font_path = f'{fury.__path__[0]}/data/files/font.png'
        image_arr = Image.open(font_path)
    else:
        if font_path is None:
            font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'

        width = num_cols_ascii*(font_size + pad*2)
        height = num_rows_ascii*(font_size + pad*2)
        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)
        for i_row, row in enumerate(chars):
            x = 2*pad
            for i_col, char in enumerate(row):
                draw.text((x, font_size*i_row + 2*pad), char, font=font)
                x += font_size
        if show:
            image.show()
        if save_path is not None:
            image.save(save_path)

        image_arr = np.array(image)
    # due vtk
    image_arr = np.flipud(image_arr)
    return image_arr, char2pos


def get_positions_labels_billboards(
        labels, centers, char2pos, scales=1):
    """This function is used to get the positions of the labels.

    Parameters
    ----------
    labels : ndarray
    centers : ndarray
    char2pos : dict
    scales : ndarray

    Returns
    -------
    labels_pad : ndarray
    labels_positions : ndarray
    uv_coordinates : ndarray
        UV texture coordinates.

    """
    labels_positions = []
    labels_pad = []
    uv_coordinates = []
    for i, (label, center) in enumerate(zip(labels, centers)):
        if isinstance(scales, list):
            scale = scales[i]
        else:
            scale = scales
        z_pad = scale*(1+0.1)
        x_pad = scale*(1+0.1)

        for i_l, l in enumerate(label):
            pad = np.array([x_pad*2*i_l, 0, z_pad])
            labels_pad.append(
              pad
            )
            labels_positions.append(center+pad)

            pos_char_begin = char2pos[l]
            mx_s = pos_char_begin[0]
            my_s = pos_char_begin[1]
            mx_e = mx_s + 1/num_cols_ascii
            my_e = my_s + 1/num_rows_ascii

            coord = np.array(
                [[[mx_s, my_e], [mx_s, my_s], [mx_e, my_s], [mx_e, my_e]]])
            uv_coordinates.append(coord)
    labels_positions = np.array(labels_positions)
    labels_pad = np.array(labels_pad)
    uv_coordinates = np.array(uv_coordinates)
    uv_coordinates = uv_coordinates.reshape(
         uv_coordinates.shape[0]*uv_coordinates.shape[2], 2).astype('float')

    return labels_pad, labels_positions, uv_coordinates
