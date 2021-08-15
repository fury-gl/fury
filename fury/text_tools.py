"""A set of tools to deal with texture atlas and freetype fonts.

The objects: TextureAtlas, TextureFont and TextureGlyph was obtained
from the example provided by Nicolas P. Rougier at here

https://github.com/rougier/freetype-py/blob/09a664d23850a7ba8cc208443b1c4f96f21116c6/examples/texture_font.py
which has the follwoing copyright

freetype-py is licensed under the terms of the new or revised BSD license, as
follows:

Copyright (c) 2011-2014, Nicolas P. Rougier
All rights reserved.

We fixed some bugs in those objects.

"""

import sys
import numpy as np
from PIL import Image
import pickle
import os
from os.path import join as pjoin

try:
    _FREETYPE_AVAILABLE = True
    import freetype as ft
except ImportError:
    _FREETYPE_AVAILABLE = False

import fury
from fury.data.fetcher import fury_home

_FONT_PATH_DEFAULT = f'{fury.__path__[0]}/data/files/font_atlas'
_FONT_PATH_TTF = f'{fury.__path__[0]}/data/files/'
_FONT_PATH_USER = pjoin(fury_home, 'font_atlas')


class TextureAtlas:
    """
    Group multiple small data regions into a larger texture.

    The algorithm is based on the article by Jukka Jylänki : "A Thousand Ways
    to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin
    Packing", February 27, 2010. More precisely, this is an implementation of
    the Skyline Bottom-Left algorithm based on C++ sources provided by Jukka
    Jylänki at: http://clb.demon.fi/files/RectangleBinPack/

    Example usage:
    --------------

    atlas = TextureAtlas(512,512,3)
    region = atlas.get_region(20,20)
    ...
    atlas.set_region(region, data)

    """

    def __init__(self, atlas_size=(1024, 1024), num_chanels=1):
        """
        Initialize a new atlas of given size.

        Parameters
        ----------

        atlas_size : tuple of int, optional
            Size of the underlying texture image. Default is (1024, 1024)
        num_chanels : int, optional
            Depth of the underlying texture

        """
        self.width = int(np.power(2, int(np.log2(atlas_size[0]) + 0.5)))
        self.height = int(np.power(2, int(np.log2(atlas_size[1]) + 0.5)))
        self.num_chanels = num_chanels
        self.nodes = [(0, 0, self.width)]
        self.data = np.zeros(
            (self.height, self.width, self.num_chanels),
            dtype=np.ubyte)
        self.used = 0

    def set_region(self, region, data):
        """
        Set a given region width provided data.

        Parameters
        ----------

        region : (int,int,int,int)
            an allocated region (x,y,width,height)

        data : numpy array
            data to be copied into given region

        """

        x, y, width, height = region
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        self.data[y:y+height, x:x+width, :] = data

    def get_region(self, width, height):
        """
        Get a free region of given size and allocate it

        Parameters
        ----------

        width : int
            Width of region to allocate

        height : int
            Height of region to allocate

        Return
        ------
            A newly allocated region as (x,y,width,height) or (-1,-1,0,0)

        """

        best_height = sys.maxsize
        best_index = -1
        best_width = sys.maxsize
        region = 0, 0, width, height

        for i in range(len(self.nodes)):
            y = self.fit(i, width, height)
            if y >= 0:
                node = self.nodes[i]
                if (y+height < best_height or
                        (y+height == best_height and node[2] < best_width)):
                    best_height = y+height
                    best_index = i
                    best_width = node[2]
                    region = node[0], y, width, height

        if best_index == -1:
            return -1, -1, 0, 0

        node = region[0], region[1]+height, width
        self.nodes.insert(best_index, node)

        i = best_index+1
        while i < len(self.nodes):
            node = self.nodes[i]
            prev_node = self.nodes[i-1]
            if node[0] < prev_node[0]+prev_node[2]:
                shrink = prev_node[0]+prev_node[2] - node[0]
                x, y, w = self.nodes[i]
                self.nodes[i] = x+shrink, y, w-shrink
                if self.nodes[i][2] <= 0:
                    del self.nodes[i]
                    i -= 1
                else:
                    break
            else:
                break
            i += 1

        self.merge()
        self.used += width*height
        return region

    def fit(self, index, width, height):
        """
        Test if region (width,height) fit into self.nodes[index]

        Parameters
        ----------

        index : int
            Index of the internal node to be tested

        width : int
            Width or the region to be tested

        height : int
            Height or the region to be tested

        """

        node = self.nodes[index]
        x, y = node[0], node[1]
        width_left = width

        if x+width > self.width:
            return -1

        i = index
        while width_left > 0:
            node = self.nodes[i]
            y = max(y, node[1])
            if y+height > self.height:
                return -1
            width_left -= node[2]
            i += 1
        return y

    def merge(self):
        """
        Merge nodes
        """

        i = 0
        while i < len(self.nodes)-1:
            node = self.nodes[i]
            next_node = self.nodes[i+1]
            if node[1] == next_node[1]:
                self.nodes[i] = node[0], node[1], node[2]+next_node[2]
                del self.nodes[i+1]
            else:
                i += 1


class TextureFont:
    """A texture font gathers a set of glyph relatively to a given font filename
    and size.

    """

    def __init__(self, atlas, filename, font_size):
        """
        Initialize font

        Parameters:
        -----------

        atlas: TextureAtlas
            Texture atlas where glyph texture will be stored

        filename: str
            Font filename

        font_size : float
            Font size

        """
        self.atlas = atlas
        self.filename = filename
        self.size = int(font_size**2)
        self.glyphs = {}
        face = ft.Face(self.filename)
        face.set_char_size(int(self.size*64))
        self._dirty = False
        metrics = face.size
        self.ascender = metrics.ascender/64.0
        self.descender = metrics.descender/64.0
        self.height = metrics.height/64.0
        self.linegap = self.height - self.ascender + self.descender
        self.num_chanels = atlas.num_chanels
        self.max_glyphy_size = np.array([0., 0.])
        try:
            ft.set_lcd_filter(ft.FT_LCD_FILTER_LIGHT)
        except ft.FT_Exception:
            pass

    def __getitem__(self, charcode):
        if charcode not in self.glyphs.keys():
            self.load('%c' % charcode)
        return self.glyphs[charcode]

    def load(self, charcodes=''):
        """
        Build glyphs corresponding to individual characters in charcodes.

        Parameters:
        -----------

        charcodes: [str | unicode]
            Set of characters to be represented

        """
        face = ft.Face(self.filename)
        pen = ft.Vector(0, 0)
        hres = 16*72
        hscale = 1.0/16

        for charcode in charcodes:
            face.set_char_size(int(self.size * 64), 0, hres, 72)
            matrix = ft.Matrix(
                int((hscale) * 0x10000), int((0.0) * 0x10000),
                int((0.0)*0x10000), int((1.0) * 0x10000))
            face.set_transform(matrix, pen)
            if charcode in self.glyphs.keys():
                continue

            self.dirty = True
            flags = ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT
            flags |= ft.FT_LOAD_TARGET_LCD

            face.load_char(charcode)
            bitmap = face.glyph.bitmap
            slot = face.glyph
            left = face.glyph.bitmap_left
            top = face.glyph.bitmap_top
            width = face.glyph.bitmap.width
            rows = face.glyph.bitmap.rows
            pitch = face.glyph.bitmap.pitch
            x, y, w, h = self.atlas.get_region(
                width/self.num_chanels+2, rows+2)
            w = int(w)
            h = int(h)
            if x < 0:
                continue
            x, y = x+1, y+1
            w, h = w-2, h-2
            data = []
            for i in range(rows):
                data.extend(bitmap.buffer[i*pitch:i*pitch+width])
            data = np.array(
                data, dtype=np.ubyte).reshape(h, w, self.atlas.num_chanels)
            gamma = 1.5
            Z = ((data/255.0)**(gamma))
            data = (Z*255).astype(np.ubyte)
            self.atlas.set_region((x, y, w, h), data)

            # Build glyph
            size = w, h
            self.max_glyphy_size[0] = max(
                self.max_glyphy_size[0], w)
            self.max_glyphy_size[1] = max(
                self.max_glyphy_size[1], h)

            offset = left, top
            advance = face.glyph.advance.x, face.glyph.advance.y

            u0 = (x + 0.0)/float(self.atlas.width)
            v0 = (y + 0.0)/float(self.atlas.height)
            u1 = (x + w - 0.0)/float(self.atlas.width)
            v1 = (y + h - 0.0)/float(self.atlas.height)
            px = w/self.atlas.width
            texcoords = (u0, v0, u1, v1)
            glyph = TextureGlyph(
                charcode, size, offset, advance, texcoords, px)
            glyph.bearingY = slot.metrics.vertBearingY/64
            glyph.metricHeight = slot.metrics.height/64
            glyph.descender = glyph.metricHeight - glyph.bearingY
            glyph.h = h
            glyph.w = w
            glyph.top = top
            glyph.st = self.size - top
            glyph.stb = glyph.st - glyph.bearingY

            glyph.ht = glyph.h - glyph.top
            self.glyphs[charcode] = glyph

            # Generate kerning
            for g in self.glyphs.values():
                # 64 * 64 because of 26.6 encoding AND the transform
                # matrix used
                # in texture_font_load_face (hres = 64)
                kerning = face.get_kerning(
                    g.charcode, charcode,
                    mode=ft.FT_KERNING_UNFITTED)
                if kerning.x != 0:
                    glyph.kerning[g.charcode] = kerning.x/(64.0*64.0)
                kerning = face.get_kerning(
                    charcode, g.charcode, mode=ft.FT_KERNING_UNFITTED)
                if kerning.x != 0:
                    g.kerning[charcode] = kerning.x/(64.0*64.0)

            # High resolution advance.x calculation
            # gindex = face.get_char_index( charcode )
            # a = face.get_advance(gindex, FT_LOAD_RENDER |
            # FT_LOAD_TARGET_LCD)/(64*72)
            # glyph.advance = a, glyph.advance[1]
        self._calc_relative_sizes()

    def _calc_relative_sizes(self):
        for char, glyph in self.glyphs.items():
            glyph.relative_size = np.array(
                [glyph.w, glyph.h])/self.max_glyphy_size
            glyph.relative_offset = np.array(glyph.offset)/self.size
            glyph.hmax = self.max_glyphy_size[1]
            if glyph.h > 0:
                glyph.pad = (glyph.stb)/glyph.h


class TextureGlyph:
    """
    A texture glyph gathers information relative to the size/offset/advance and
    texture coordinates of a single character. It is generally built
    automatically by a TextureFont.

    """

    def __init__(self, charcode, size, offset, advance, texcoords, px):
        """
        Build a new texture glyph

        Parameter:
        ----------

        charcode : char
            Represented character

        size: tuple of 2 ints
            Glyph size in pixels

        offset: tuple of 2 floats
            Glyph offset relatively to anchor point

        advance: tuple of 2 floats
            Glyph advance

        texcoords: tuple of 4 floats
            Texture coordinates of bottom-left and top-right corner
            of glyph bounding box

        """
        self.charcode = charcode
        self.size = size
        self.relative_size = np.array([1., 1.])
        self.px = px
        self.offset = offset
        self.advance = advance
        self.texcoords = texcoords
        self.kerning = {}

    def get_kerning(self, charcode):
        """ Get kerning information

        Parameters:
        -----------

        charcode: char
            Character preceding this glyph
        """
        if charcode in self.kerning.keys():
            return self.kerning[charcode]
        else:
            return 0


def list_fonts_available(fullpath=False):
    """ List available fonts in the system

    Parameters
    ----------
    fullpath: bool, optional
        If True, return full path, otherwise, return only the font name

    Returns
    -------
    fonts: list or dict
        Font names available in your FURY installation.
        A dictionary with full path and font names is returned if fullpath is
        True.

    """

    fonts = {}
    for f in os.listdir(_FONT_PATH_DEFAULT):
        fonts[f] = f'{_FONT_PATH_DEFAULT}/{f}/'

    if not os.path.exists(_FONT_PATH_USER):
        return fonts
    for f in os.listdir(_FONT_PATH_USER):
        fonts[f] = f'{_FONT_PATH_USER}/{f}/'
    if not fullpath:
        return list(fonts.keys())

    return fonts


def create_atlas_font(
        name, font_path, font_size_res=7,
        atlas_size=(1024, 1024),
        show=False, use_system_path=False):
    """This function is used to create a bitmap font.

    Parameters
    ----------
    name : str
        Name of the font to be saved
    font_size_res : int
        The size of the font.
    font_path : str
        The path to the font file.
    pad : int
        The padding of the font.
    atlas_size : tuple, optional
        The size of the texture atlas in pixels.
    show : bool
        Whether to show the result.
    use_system_path : bool, optional
        If True, the font path is the system path, otherwise, it is the
        user path.

    Returns
    -------
    image_array : ndarray
        The image array.
    char2pos : dict
        A dictionary that maps characters to their positions in the
        numpy array.

    """
    if not _FREETYPE_AVAILABLE:
        raise ImportError('Pleasse, install  the freetype-py lib')

    font_path_save = _FONT_PATH_DEFAULT if use_system_path else _FONT_PATH_USER
    folder = font_path_save + f'/{name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(
            f'Font {name} already exists. ' +
            'Please choose a another name.')
        return folder

    texture_atlas = TextureAtlas(num_chanels=1, atlas_size=atlas_size)
    image_arr = texture_atlas.data

    image_arr = texture_atlas.data.reshape(
        (texture_atlas.data.shape[0], texture_atlas.data.shape[1]))
    texture_font = TextureFont(
        texture_atlas,
        font_path,
        font_size=font_size_res)
    ascii_chars = ''.join([chr(i) for i in range(32, 127)])
    texture_font.load(ascii_chars)
    char2coord = {
        c: glyph
        for c, glyph in texture_font.glyphs.items()
    }

    if show:
        image = Image.fromarray(image_arr).convert('P')
        image.show()

    image = Image.fromarray(image_arr).convert('P')
    image.save(folder + '/atlas.bmp')
    pickle.dump(char2coord, open(folder+'/char2coord.p', 'wb'))
    # due vtk
    return folder


def _create_fury_system_atlas_fonts(font_size_res=12, atlas_size=(1024, 1024)):
    """This function is used to create all the atlas fonts
    in the system. Using the TTF fonts available in the fury files
    folder.
    """

    # list all TTF files in a folder
    fonts = [f for f in os.listdir(_FONT_PATH_TTF) if f.endswith('.ttf')]
    for font in fonts:
        font_name = font.split('.')[0].replace(' ', '').replace('-', '_')
        # create a font atlas
        print('Creating system font atlas:', font)
        create_atlas_font(
            font_name,
            f'{_FONT_PATH_TTF}/{font}',
            font_size_res=font_size_res,
            atlas_size=atlas_size,
            use_system_path=True,
            show=False)


def get_texture_atlas_font(font_name='FreeMono'):
    """This function is used to create a bitmap font.

    Parameters
    ----------
    font_name : str
        Name of the font to be loaded.
    Returns
    -------
    image_array : ndarray
        The image array.
    char2pos : dict
        A dictionary that maps characters to their positions in the
        numpy array.

    """
    fonts_available = list_fonts_available(fullpath=True)
    if font_name not in fonts_available.keys():
        raise ValueError(
            "Font name %s not available. "
            "Please choose one of the following fonts: %s" % (
                font_name, list(fonts_available.keys())))
    font_path = fonts_available[font_name]
    image_arr = Image.open(font_path + 'atlas.bmp')
    char2coord = pickle.load(open(font_path + 'char2coord.p', 'rb'))

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
    relative_sizes = []
    for i, (label, center) in enumerate(zip(labels, centers)):
        if isinstance(scales, list):
            scale = scales[i]
        else:
            scale = scales

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

        sum_x_spacing = 0
        for i_l, char in enumerate(label):
            if char not in char2coord.keys():
                char = '?'
            glyph = char2coord[char]

            relative_size = glyph.relative_size
            rx = relative_size[0]
            relative_sizes.append(relative_size)
            pad = np.array([0., 0, 0], dtype='float64')

            if glyph.h > 0:
                offset = scale*y_offset_ratio/relative_size[1]
                pad[1] -= scale*glyph.pad - offset

            pad_x = (scale*x_offset_ratio)
            if rx == 0:
                rx = 1
            pad[0] = (sum_x_spacing + pad_x + align_pad)/rx  # + align_pad
            sum_x_spacing += pad_x
            labels_pad.append(
              pad
            )
            labels_positions.append(center)

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
    relative_sizes = np.repeat(np.array(relative_sizes), 4, axis=0)
    uv_coordinates = uv_coordinates.reshape(
         uv_coordinates.shape[0]*uv_coordinates.shape[2], 2).astype('float')

    return labels_pad, labels_positions, uv_coordinates, relative_sizes
