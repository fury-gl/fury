"""Module that provide actors to render."""

import warnings
import os.path as op
import numpy as np
import vtk

from fury.shaders import load, shader_to_actor, attribute_to_actor
from fury.utils import rgb_to_vtk, get_actor_from_primitive
import fury.primitive as fp
from fury import text_tools


def bitmap_labels(
        centers,
        labels,
        colors=(0, 1, 0),
        scales=1,
        font_size=50,
        font_path=None,
        ):
    """Create a bitmap label actor that always faces the camera.

    Parameters
    ----------
    centers : ndarray, shape (N, 3)
    labels  : list
        list of strings
    colors : ndarray (N,3) or (N, 4) or tuple (3,) or tuple (4,)
        RGB or RGBA (for opacity) R, G, B and A should be at the range [0, 1]
    scales : ndarray, optional
        shape (N) or (N,3) or float or int, optional
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
            labels, centers, char2pos, scales)

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

    img_vtk = rgb_to_vtk(np.ascontiguousarray(img_arr))
    tex = vtk.vtkTexture()
    tex.SetInputDataObject(img_vtk)
    tex.Update()
    sq_actor.GetProperty().SetTexture('charactersTexture', tex)
    attribute_to_actor(
        sq_actor,
        uv,
        'vUV')
    padding = np.repeat(padding, 4, axis=0)
    # num_labels = padding.shape[0]
    # padding = np.repeat(np.array([padding]).T, 6, axis=0).reshape(num_labels*6, 3)
    # print(padding[0:10])

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