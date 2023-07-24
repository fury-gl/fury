import numpy as np
from fury.actor import Actor
from fury.io import load_image
from fury.lib import Texture, WindowToImageFilter
from fury.utils import rgb_to_vtk
from fury.window import RenderWindow


WRAP_MODE_DIC = {"clamptoedge" : Texture.ClampToEdge,
                 "repeat" : Texture.Repeat,
                 "mirroredrepeat" : Texture.MirroredRepeat,
                 "clamptoborder" : Texture.ClampToBorder}

BLENDING_MODE_DIC = {"none" : 0, "replace" : 1,
                     "modulate" : 2, "add" : 3,
                     "addsigned" : 4, "interpolate" : 5,
                     "subtract" : 6}

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
    """Capture a rendered window and pass it as a texture to the given actor.
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

    windowToImageFilter = WindowToImageFilter()
    windowToImageFilter.SetInput(window)

    windowToImageFilter.Update()

    texture = Texture()
    texture.SetMipmap(True)
    texture.SetInputConnection(windowToImageFilter.GetOutputPort())
    texture.SetBorderColor(*border_color)
    texture.SetWrap(WRAP_MODE_DIC[wrap_mode.lower()])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(BLENDING_MODE_DIC[blending_mode.lower()])

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
    """Pass an imported texture to an actor.
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
    
    texture = Texture()

    colormapArray = load_image(path_to_texture)
    colormapData = rgb_to_vtk(colormapArray)

    texture.SetInputDataObject(colormapData)
    texture.SetBorderColor(*border_color)
    texture.SetWrap(WRAP_MODE_DIC[wrap_mode.lower()])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(BLENDING_MODE_DIC[blending_mode.lower()])

    target_actor.GetProperty().SetTexture(texture_name, texture)


def colormap_to_texture(
        colormap : np.array,
        texture_name : str,
        target_actor : Actor,
        interpolate : bool = True):
    """Convert a colormap to a texture and pass it to an actor.
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