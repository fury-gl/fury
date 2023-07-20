import numpy as np
from fury.window import RenderWindow
from fury.actor import Actor
from fury.lib import Texture, WindowToImageFilter
from fury.io import load_image
from fury.utils import rgb_to_vtk


def window_to_texture(
        window : RenderWindow,
        texture_name : str,
        target_actor : Actor,
        blending_mode : str = "None",
        wrap_mode : str = "ClampToBorder",
        border_color : tuple = (
            0.0,
            0.0,
            0.0,
            1.0),
        interpolate : bool = True):
    """Captures a rendered window and pass it as a texture to the given actor.
    Parameters
    ----------
    window : window.RenderWindow
        Window to be captured.
    texture_name : str
        Name of the texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the texture.
    blending_mode : str, optional
        Texture blending mode. The options are:
    1. None
    2. Replace
    3. Modulate
    4. Add
    5. AddSigned
    6. Interpolate
    7. Subtract
    wrap_mode : str, optional
        Texture wrapping mode. The options are:
    1. ClampToEdge
    2. Repeat
    3. MirroredRepeat
    4. ClampToBorder
    border_color : tuple (4, ), optional
        Texture RGBA border color.
    interpolate : bool, optional
        Texture interpolation."""

    wrap_mode_dic = {"ClampToEdge" : Texture.ClampToEdge,
                     "Repeat" : Texture.Repeat,
                     "MirroredRepeat" : Texture.MirroredRepeat,
                     "ClampToBorder" : Texture.ClampToBorder}

    blending_mode_dic = {"None" : 0, "Replace" : 1,
                         "Modulate" : 2, "Add" : 3,
                         "AddSigned" : 4, "Interpolate" : 5,
                         "Subtract" : 6}

    r, g, b, a = border_color

    windowToImageFilter = WindowToImageFilter()
    windowToImageFilter.SetInput(window)

    windowToImageFilter.Update()

    texture = Texture()
    texture.SetInputConnection(windowToImageFilter.GetOutputPort())
    texture.SetBorderColor(r, g, b, a)
    texture.SetWrap(wrap_mode_dic[wrap_mode])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(blending_mode_dic[blending_mode])

    target_actor.GetProperty().SetTexture(texture_name, texture)


def texture_to_actor(
        path_to_texture : str,
        texture_name : str,
        target_actor : Actor,
        blending_mode : str = "None",
        wrap_mode : str = "ClampToBorder",
        border_color : tuple = (
            0.0,
            0.0,
            0.0,
            1.0),
        interpolate : bool = True):
    """Passes an imported texture to an actor.
    Parameters
    ----------
    path_to_texture : str
        Texture image path.
    texture_name : str
        Name of the texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the texture.
    blending_mode : str
        Texture blending mode. The options are:
    1. None
    2. Replace
    3. Modulate
    4. Add
    5. AddSigned
    6. Interpolate
    7. Subtract
    wrap_mode : str
        Texture wrapping mode. The options are:
    1. ClampToEdge
    2. Repeat
    3. MirroredRepeat
    4. ClampToBorder
    border_color : tuple (4, )
        Texture RGBA border color.
    interpolate : bool
        Texture interpolation."""

    wrap_mode_dic = {"ClampToEdge" : Texture.ClampToEdge,
                     "Repeat" : Texture.Repeat,
                     "MirroredRepeat" : Texture.MirroredRepeat,
                     "ClampToBorder" : Texture.ClampToBorder}

    blending_mode_dic = {"None" : 0, "Replace" : 1,
                         "Modulate" : 2, "Add" : 3,
                         "AddSigned" : 4, "Interpolate" : 5,
                         "Subtract" : 6}

    r, g, b, a = border_color

    texture = Texture()

    colormapArray = load_image(path_to_texture)
    colormapData = rgb_to_vtk(colormapArray)

    texture.SetInputDataObject(colormapData)
    texture.SetBorderColor(r, g, b, a)
    texture.SetWrap(wrap_mode_dic[wrap_mode])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(blending_mode_dic[blending_mode])

    target_actor.GetProperty().SetTexture(texture_name, texture)


def colormap_to_texture(
        colormap : np.array,
        texture_name : str,
        target_actor : Actor,
        interpolate : bool = True):
    """Converts a colormap to a texture and pass it to an actor.
    Parameters
    ----------
    colormap : np.array (N, 4) or (1, N, 4)
        RGBA color map array. The array can be two dimensional, although a three dimensional one is preferred.
    texture_name : str
        Name of the color map texture to be passed to the actor.
    target_actor : Actor
        Target actor to receive the color map texture.
    interpolate : bool
        Color map texture interpolation."""

    if len(colormap.shape) == 2:
        colormap = np.array([colormap])

    texture = Texture()

    cmap = (255*colormap).astype(np.uint8)
    cmap = rgb_to_vtk(cmap)

    texture.SetInputDataObject(cmap)
    texture.SetWrap(Texture.ClampToEdge)
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(0)

    target_actor.GetProperty().SetTexture(texture_name, texture)