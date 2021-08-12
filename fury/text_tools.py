import numpy as np
from PIL import Image
import pickle
import fury
from fury.texture_font import TextureAtlas, TextureFont


def create_bitmap_font(
        font_size=50, font_path=None,
        show=False, save_path=None):
    """This function is used to create a bitmap font.

    Parameters
    ----------
    font_size : int
        The size of the font.
    font_path : str
        The path to the font file.
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

    if font_size == 50 and font_path is None:
        font_path = f'{fury.__path__[0]}/data/files/font'
        image_arr = Image.open(font_path+'.bmp')
        char2coord = pickle.loads(open(font_path + '_char2coord.p', 'rb').read())
    else:
        if font_path is None:
            font_path = f'{fury.__path__[0]}/data/files/FreeMono.ttf'
        texture_atlas = TextureAtlas(depth=1)
        image_arr = texture_atlas.data

        image_arr = texture_atlas.data.reshape(
            (texture_atlas.data.shape[0], texture_atlas.data.shape[1]))
        texture_font = TextureFont(texture_atlas, font_path, font_size)
        ascii_chars = ''.join([chr(i) for i in range(32, 127)])
        texture_font.load(ascii_chars)
        char2coord = {
            c: glyph
            for c, glyph in texture_font.glyphs.items()
        }

        if show:
            image = Image.fromarray(image_arr).convert('P')
            image.show()
        if save_path is not None:
            image = Image.fromarray(image_arr).convert('P')
            image.save(save_path + '.bmp')
            pickle.dump(char2coord, open(save_path + '_char2coord.p', 'wb'))
    # due vtk
    image_arr = np.flipud(image_arr)
    return image_arr, char2coord


def get_positions_labels_billboards(
        labels, centers, char2coord, scales=1,
        align='center',
        x_offset_ratio=1, y_offset_ratio=1,):
    """This function is used to get the positions of the labels.

    Parameters
    ----------
    labels : ndarray
    centers : ndarray
    char2pos : dict
    scales : ndarray
    align : str, {'left', 'right', 'center'}
    x_offset_ratio : float
        Percentage of the width to offset the labels on the x axis.
    y_offset_ratio : float
        Percentage of the height to offset the labels on the y axis.

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
        y_pad = scale*y_offset_ratio
        x_pad = scale*x_offset_ratio
        align_pad = 0.
        if align == 'left':
            align_pad = 0
        elif align == 'right':
            align_pad = -x_pad*len(label)
            align_pad += x_pad
        elif align == 'center':
            align_pad = -x_pad*len(label)
            if not len(label) % 2 == 0:
                align_pad += x_pad
            align_pad /= 2
        for i_l, char in enumerate(label):
            pad = np.array([x_pad*i_l + align_pad, y_pad, 0])
            labels_pad.append(
              pad
            )
            labels_positions.append(center)
            if char not in char2coord.keys():
                char = '?'
            glyph = char2coord[char]
            mx_s = glyph.texcoords[0]
            my_s = glyph.texcoords[1]
            mx_e = glyph.texcoords[2]
            my_e = glyph.texcoords[3]
            coord = np.array(
                [[[mx_s, my_e], [mx_s, my_s], [mx_e, my_s], [mx_e, my_e]]])
            uv_coordinates.append(coord)
    labels_positions = np.array(labels_positions)
    labels_pad = np.array(labels_pad)
    uv_coordinates = np.array(uv_coordinates)
    uv_coordinates = uv_coordinates.reshape(
         uv_coordinates.shape[0]*uv_coordinates.shape[2], 2).astype('float')

    return labels_pad, labels_positions, uv_coordinates
