"""Module that provide actors to render."""

import warnings
import os.path as op
import numpy as np
import vtk

from fury.shaders import load, shader_to_actor, attribute_to_actor
from fury.utils import (
    rgb_to_vtk, get_actor_from_primitive,  one_chanel_to_vtk)
import fury.primitive as fp
from fury import text_tools


def bitmap_labels(
        centers,
        labels,
        colors=(0, 1, 0),
        scales=1,
        align='center',
        x_offset_ratio=1,
        y_offset_ratio=1,
        font_size=50,
        font_path=None,
        ):
    """Create a bitmap label actor that always faces the camera.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
    labels  : list
        list of strings
    colors : array or ndarray
    scales : float
    align : str, {left, right, center}
    x_offset_ratio : float
        Percentage of the width to offset the labels on the x axis.
    y_offset_ratio : float
        Percentage of the height to offset the labels on the y axis.
    font_size : int, optional
        size of the text
    font_path : str, optional
        str of path to font file

    Returns
    -------
    vtkActor

    """
    img_arr, char2pos = text_tools.create_bitmap_font(
        font_size=font_size, font_path=font_path, show=False)
    padding, labels_positions, uv = text_tools.get_positions_labels_billboards(
            labels, centers, char2pos, scales,
            align=align,
            x_offset_ratio=x_offset_ratio, y_offset_ratio=y_offset_ratio)
    # num_chars = labels_positions.shape[0]
    verts, faces = fp.prim_square()
    res = fp.repeat_primitive(
        verts, faces, centers=labels_positions, colors=colors,
        scales=scales)

    big_verts, big_faces, big_colors, big_centers = res
    sq_actor = get_actor_from_primitive(big_verts, big_faces, big_colors)
    sq_actor.GetMapper().SetVBOShiftScaleMethod(False)
    sq_actor.GetProperty().BackfaceCullingOff()

    attribute_to_actor(sq_actor, big_centers, 'center')

    vs_dec_code = load("billboard_dec.vert")
    vs_dec_code += f'\n{load("text_billboard_dec.vert")}'

    vs_impl_code = load("text_billboard_impl.vert")

    fs_dec_code = load('billboard_dec.frag')
    fs_dec_code += f'\n{load("text_billboard_dec.frag")}'

    fs_impl_code = load('billboard_impl.frag')
    fs_impl_code += f'\n{load("text_billboard_impl.frag")}'

    img_vtk = one_chanel_to_vtk(img_arr)
    tex = vtk.vtkTexture()
    tex.SetInputDataObject(img_vtk)
    tex.Update()
    sq_actor.GetProperty().SetTexture('charactersTexture', tex)
    attribute_to_actor(
        sq_actor,
        uv,
        'vUV')
    padding = np.repeat(padding, 4, axis=0)
    attribute_to_actor(
        sq_actor,
        padding,
        'vPadding')

    shader_to_actor(sq_actor, "vertex", impl_code=vs_impl_code,
                    decl_code=vs_dec_code)
    shader_to_actor(sq_actor, "fragment", decl_code=fs_dec_code)
    shader_to_actor(sq_actor, "fragment", impl_code=fs_impl_code,
                    block="light")

    return sq_actor
