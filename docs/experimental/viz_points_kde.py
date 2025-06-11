import numpy as np

from fury import actor, window
from fury.colormap import create_colormap
from fury.lib import Texture, WindowToImageFilter
from fury.shaders import shader_apply_effects
from fury.utils import rgb_to_vtk


def window_to_texture(
        window : window.RenderWindow,
        texture_name : str,
        target_actor : actor.Actor,
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
    target_actor : actor.Actor
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

    wrap_mode_dic = {"clamptoedge" : Texture.ClampToEdge,
                     "repeat" : Texture.Repeat,
                     "mirroredrepeat" : Texture.MirroredRepeat,
                     "clamptoborder" : Texture.ClampToBorder}

    blending_mode_dic = {"none" : 0, "replace" : 1,
                         "modulate" : 2, "add" : 3,
                         "addsigned" : 4, "interpolate" : 5,
                         "subtract" : 6}

    windowToImageFilter = WindowToImageFilter()
    windowToImageFilter.SetInput(window)

    windowToImageFilter.Update()

    texture = Texture()
    texture.SetInputConnection(windowToImageFilter.GetOutputPort())
    texture.SetBorderColor(*border_color)
    texture.SetWrap(wrap_mode_dic[wrap_mode.lower()])
    texture.SetInterpolate(interpolate)
    texture.MipmapOn()
    texture.SetBlendingMode(blending_mode_dic[blending_mode.lower()])

    target_actor.GetProperty().SetTexture(texture_name, texture)


def colormap_to_texture(
        colormap : np.array,
        texture_name : str,
        target_actor : actor.Actor,
        interpolate : bool = True):
    """Convert a colormap to a texture and pass it to an actor.

    Parameters
    ----------
    colormap : np.array (N, 4) or (1, N, 4)
        RGBA color map array. The array can be two dimensional, although a three dimensional one is preferred.
    texture_name : str
        Name of the color map texture to be passed to the actor.
    target_actor : actor.Actor
        Target actor to receive the color map texture.
    interpolate : bool, optional
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


def shader_custom_uniforms(actor : actor.Actor, shader_type : str):
    """Ease the passing of uniform values to the shaders by returning ``actor.GetShaderProperty().GetVertexCustomUniforms()``,
    that give access to the ``SetUniform`` methods.
    Parameters
    ----------
    actor : actor.Actor
          Actor which the uniform values will be passed to.
    shader_type : str
          Shader type of the uniform values to be passed. It can be:
          * "vertex"
          * "fragment"
          * "geometry"
    """
    if shader_type.lower() == "vertex":
        return actor.GetShaderProperty().GetVertexCustomUniforms()
    elif shader_type.lower() == "fragment":
        return actor.GetShaderProperty().GetFragmentCustomUniforms()
    elif shader_type.lower() == "geometry":
        return actor.GetShaderProperty().GetGeometryCustomUniforms()
    else:
        raise ValueError("Shader type unknown.")


def normalize(array : np.array, min : float = 0.0, max : float = 1.0):
    """Convert an array to a given desired range.

    Parameters
    ----------
    array : np.ndarray
        Array to be normalized.
    min : float, optional
        Bottom value of the interval of normalization. If no value is given, it is passed as 0.0.
    max : float, optional
        Upper value of the interval of normalization. If no value is given, it is passed as 1.0.

    Returns
    -------
    array : np.array
        Array converted to the given desired range.
    """
    if np.max(array) != np.min(array):
        return ((array - np.min(array))/(np.max(array) - np.min(array)))*(max - min) + min
    else:
        raise ValueError(
            "Can't normalize an array which maximum and minimum value are the same.")


kde_dec = """
float kde(vec3 point, float sigma){
    return exp(-1.0*pow(length(point), 2.0)/(2.0*sigma*sigma) );
}
"""

kde_impl = """
float current_kde = kde(normalizedVertexMCVSOutput, sigma);
color = vec3(current_kde);
fragOutput0 = vec4(color, 1.0);
"""

tex_dec = """
vec3 color_mapping(float intensity, sampler2D colormapTexture){;
    return texture(colormapTexture, vec2(intensity, 0)).rgb;
}
"""

tex_impl = """
vec2 renorm_tex = normalizedVertexMCVSOutput.xy*0.5 + 0.5;
float intensity = texture(screenTexture, renorm_tex).r;

if(intensity<=0.0){
    discard;
}else{
    color = color_mapping(intensity, colormapTexture).rgb;
    fragOutput0 = vec4(color, 1.0);
}
"""


# Windows and scenes setup
width, height = (1920, 1080)
offWidth, offHeight = (1080, 1080)

offScene = window.Scene()

off_manager = window.ShowManager(
    offScene,
    "demo",
    (offWidth,
     offHeight),
    reset_camera=True,
    order_transparent=True)

off_manager.window.SetOffScreenRendering(True)

off_manager.initialize()


scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 0.0))

manager = window.ShowManager(
    scene,
    "demo",
    (width,
     height),
    reset_camera=True,
    order_transparent=True)


manager.initialize()

n_points = 1000
points = np.random.rand(n_points, 3)
points = normalize(points, -5, 5)
sigma = 0.25
scale = 0.5

billboard = actor.billboard(
    points,
    (0.0,
     0.0,
     1.0),
    scales=scale,
    fs_dec=kde_dec,
    fs_impl=kde_impl)

# Blending and uniforms setup
shader_apply_effects(off_manager.window, billboard, window.gl_disable_depth)
shader_apply_effects(off_manager.window, billboard, window.gl_set_additive_blending)
shader_custom_uniforms(billboard, "fragment").SetUniformf("sigma", sigma)

off_manager.scene.add(billboard)

off_manager.render()


# Render to second billboard for color map post-processing.
textured_billboard = actor.billboard(np.array([[0.0, 0.0, 0.0]]), (1.0, 1.0, 1.0),
                                     scales=3.0, fs_dec=tex_dec, fs_impl=tex_impl)

cmap = create_colormap(np.arange(0.0, 1.0, 1/256), "viridis")

colormap_to_texture(cmap, "colormapTexture", textured_billboard)


window_to_texture(
    off_manager.window,
    "screenTexture",
    textured_billboard,
    blending_mode="Interpolate")

manager.scene.add(textured_billboard)

interactive = False

if interactive:
    manager.start()

window.record(scene, size = (800, 800), out_path = "points_kde.png")