import numpy as np
from fury import window, actor
from fury.shaders import compose_shader, shader_apply_effects, import_fury_shader, shader_custom_uniforms
from os.path import join
from fury.postprocessing import window_to_texture, colormap_to_texture
from matplotlib import colormaps


def normalize(array : np.array, min : float = 0.0, max : float = 1.0, axis : int = 0):
    """Converts an array to a given desired range.

    Parameters
    ----------
    array : np.ndarray
    Array to be normalized.
    min : float
    Bottom value of the interval of normalization. If no value is given, it is passed as 0.0.
    max : float
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


kde_dec = import_fury_shader(join("utils", "normal_distribution.glsl"))

kde_impl = """
float current_kde = kde(normalizedVertexMCVSOutput, sigma);
color = vec3(current_kde);
fragOutput0 = vec4(color, 1.0);
"""

tex_dec = import_fury_shader(join("effects", "color_mapping.glsl"))

tex_impl = """
// Turning screen coordinates to texture coordinates
vec2 renorm_tex = normalizedVertexMCVSOutput.xy*0.5 + 0.5;
float intensity = texture(screenTexture, renorm_tex).r;

if(intensity<=0.0){
    discard;
}else{
    color = color_mapping(intensity, colormapTexture).rgb;
    fragOutput0 = vec4(color, 1.0);
}
"""


fs_dec = compose_shader([kde_dec])

fs_impl = compose_shader([kde_impl])


# Windows and scenes setup
width, height = (1920, 1080)
offWidth, offHeight = (1080, 1080)

offScene = window.Scene()
offScene.set_camera(position=(-6, 5, -10),
                    focal_point=(0.0,
                              0.0,
                              0.0),
                    view_up=(0.0, 0.0, 0.0))

off_manager = window.ShowManager(
    offScene,
    "demo",
    (offWidth,
     offHeight))

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
     height))


manager.initialize()

n_points = 1000
points = np.random.rand(n_points, 3)
points = normalize(points, -5, 5)
sigma = 0.3
scale = 0.5

billboard = actor.billboard(
    points,
    (0.0,
     0.0,
     1.0),
    scales=scale,
    fs_dec=fs_dec,
    fs_impl=fs_impl)

# Blending and uniforms setup
shader_apply_effects(off_manager.window, billboard, window.gl_disable_depth)
shader_apply_effects(off_manager.window, billboard, window.gl_set_additive_blending)
shader_custom_uniforms(billboard, "fragment").SetUniformf("sigma", sigma)

off_manager.scene.add(billboard)

off_manager.render()

scale = np.array([width/height, 1.0, 0.0])

# Render to second billboard for color map post-processing.
textured_billboard = actor.billboard(np.array([[0.0, 0.0, 0.0]]), (1.0, 1.0, 1.0),
                                     scales=10.0, fs_dec=tex_dec, fs_impl=tex_impl)

# Disables the texture warnings
textured_billboard.GetProperty().GlobalWarningDisplayOff() 

cmap = colormaps["inferno"]
cmap = np.array([cmap(i) for i in np.arange(0.0, 1.0, 1/256)])

colormap_to_texture(cmap, "colormapTexture", textured_billboard)


def event_callback(obj, event):
    pos, focal, vu = manager.scene.get_camera()
    off_manager.scene.set_camera(pos, focal, vu)
    off_manager.scene.Modified()
    off_manager.render()

    window_to_texture(
    off_manager.window,
    "screenTexture",
    textured_billboard,
    blending_mode="Interpolate")

    

window_to_texture(
    off_manager.window,
    "screenTexture",
    textured_billboard,
    blending_mode="Interpolate")

manager.scene.add(textured_billboard)

manager.add_iren_callback(event_callback, "RenderEvent")

manager.start()
