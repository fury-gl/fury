import numpy as np
import numpy.testing as npt
import os

from fury.actor import (billboard, 
                        cube)
from fury.actors.effect_manager import (colormap_to_texture, 
                                        EffectManager, 
                                        texture_to_actor,
                                        window_to_texture)
from fury.colormap import create_colormap
from fury.lib import Texture
from fury.shaders import shader_custom_uniforms, import_fury_shader
from fury.window import (Scene, 
                         ShowManager, 
                         record)
from fury import window

width, height = (400, 400)

WRAP_MODE_DIC = {"clamptoedge" : Texture.ClampToEdge,
                 "repeat" : Texture.Repeat,
                 "mirroredrepeat" : Texture.MirroredRepeat,
                 "clamptoborder" : Texture.ClampToBorder}

BLENDING_MODE_DIC = {"none" : 0, "replace" : 1,
                     "modulate" : 2, "add" : 3,
                     "addsigned" : 4, "interpolate" : 5,
                     "subtract" : 6}

points = np.array([[0.36600749, 0.65827962, 0.53083986],
                    [0.97657922, 0.78730041, 0.13946709],
                    [0.7441061 , 0.26322696, 0.8683115 ],
                    [0.14606987, 0.05490296, 0.98723486],
                    [0.71673873, 0.29188497, 0.02825102],
                    [0.90364963, 0.06387054, 0.91557011],
                    [0.11106939, 0.73972495, 0.49771819],
                    [0.63509055, 0.26659524, 0.4790886 ],
                    [0.20590893, 0.56012136, 0.78304187],
                    [0.30247726, 0.28023438, 0.6883304 ],
                    [0.58971475, 0.67312749, 0.47656539],
                    [0.26257592, 0.23964672, 0.64210249],
                    [0.26631165, 0.35701288, 0.88390072],
                    [0.01108113, 0.87276217, 0.99321825],
                    [0.68792169, 0.42725589, 0.92290326],
                    [0.09702907, 0.69950028, 0.97210289],
                    [0.86744636, 0.29614399, 0.2729772 ],
                    [0.77511449, 0.6912353 , 0.97596621],
                    [0.5919642 , 0.25713794, 0.0692452 ],
                    [0.47674521, 0.94254354, 0.71231971],
                    [0.50177591, 0.19320157, 0.91493713],
                    [0.27073903, 0.58171665, 0.79582017],
                    [0.76282237, 0.35119548, 0.80971555],
                    [0.43065933, 0.87678895, 0.57491155],
                    [0.34213045, 0.70619672, 0.43970999],
                    [0.38793158, 0.33048163, 0.91679507],
                    [0.68375111, 0.47934201, 0.86197378],
                    [0.67829585, 0.80616031, 0.76974334],
                    [0.01784785, 0.24857252, 0.89913317],
                    [0.8458996,  0.51551657, 0.69597985]])

sigmas = np.array([[0.56193862], [0.1275334 ], [0.91069059],
                    [0.01177131], [0.67799239], [0.95772282],
                    [0.55834784], [0.60151661], [0.25946789],
                    [0.88343075], [0.24011991], [0.05879632],
                    [0.6370561 ], [0.23859789], [0.18654873],
                    [0.70008281], [0.02968318], [0.01304724],
                    [0.08251756], [0.625351  ], [0.89982588],
                    [0.62378987], [0.8661594 ], [0.05583442],
                    [0.60157791], [0.84737657], [0.36433019],
                    [0.13263502], [0.30937519], [0.88979053]])


def test_window_to_texture(interactive = False):
    fs_impl = """
    vec2 res_factor = vec2(res.y/res.x, 1.0);
    vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
    vec4 tex = texture(screenTexture, renorm_tex);
    fragOutput0 = vec4(tex);
    """

    on_scene = window.Scene()
    on_scene.set_camera(position=(-2, 1, -2),
                    focal_point=(0.0,
                                 0.0,
                                 0.0),
                    view_up=(0.0, 0.0, 0.0))

    on_manager = window.ShowManager(
        on_scene,
        "on_manager",
        (width,
        height))

    off_scene = window.Scene()
    off_scene.set_camera(position=(-4, 2, -4),
                    focal_point=(0.0,
                                 0.0,
                                 0.0),
                    view_up=(0.0, 0.0, 0.0))

    off_manager = window.ShowManager(
        off_scene,
        "off_manager",
        (width,
        height))
    
    off_manager.window.SetOffScreenRendering(True)
    off_manager.initialize()

    scale = np.array([[width/height, 1.0, 0.0]])
    c = cube(np.array([[0.0, 0.0, 0.0]]), colors=(1.0, 1.0, 0.0))
    bill = billboard(np.array([[0.0, 0.0, 0.0]]), scales = scale, fs_impl=fs_impl)
    shader_custom_uniforms(bill, "fragment").SetUniform2f("res", off_manager.size)

    off_manager.scene.add(c)
    on_manager.scene.add(bill)

    tex_name = "screenTexture"
    blending_mode = "Interpolate"
    dtype = "RGBA"
    wrap_mode = "ClampToBorder"
    border_color = (0.0, 0.0, 0.0, 1.0)
    interpolate = True

    off_manager.render()
    window_to_texture(off_manager.window, 
                      tex_name, 
                      bill, 
                      blending_mode=blending_mode, 
                      wrap_mode=wrap_mode, 
                      border_color=border_color,
                      interpolate=interpolate,
                      d_type=dtype)
    
    if interactive:
        window.show(on_manager.scene)

    n_textures = bill.GetProperty().GetNumberOfTextures()
    texture = bill.GetProperty().GetTexture(tex_name)

    npt.assert_equal(1, n_textures)
    npt.assert_equal(WRAP_MODE_DIC[wrap_mode.lower()], texture.GetWrap())
    npt.assert_equal(BLENDING_MODE_DIC[blending_mode.lower()], texture.GetBlendingMode())
    npt.assert_array_almost_equal(list(border_color), texture.GetBorderColor())
    npt.assert_equal(interpolate, texture.GetInterpolate())
    npt.assert_equal(True, texture.GetMipmap())

    record(on_manager.scene, out_path="test_window.png", size=(width, height))

def test_texture_to_actor(interactive = False):
    fs_impl = """
    vec2 res_factor = vec2(res.y/res.x, 1.0);
    vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
    vec4 tex = texture(screenTexture, renorm_tex);
    fragOutput0 = vec4(tex);
    """

    on_scene = window.Scene()
    on_scene.set_camera(position=(-2, 1, -2),
                    focal_point=(0.0,
                                 0.0,
                                 0.0),
                    view_up=(0.0, 0.0, 0.0))

    on_manager = window.ShowManager(
        on_scene,
        "on_manager",
        (width,
        height))

    scale = np.array([[width/height, 1.0, 0.0]])
    bill = billboard(np.array([[0.0, 0.0, 0.0]]), scales = scale, fs_impl=fs_impl)
    shader_custom_uniforms(bill, "fragment").SetUniform2f("res", on_manager.size)

    on_manager.scene.add(bill)

    tex_name = "screenTexture"
    blending_mode = "Interpolate"
    dtype = "RGBA"
    wrap_mode = "ClampToBorder"
    border_color = (0.0, 0.0, 0.0, 1.0)
    interpolate = True

    texture_to_actor("test_window.png",
                      tex_name, 
                      bill, 
                      blending_mode=blending_mode, 
                      wrap_mode=wrap_mode, 
                      border_color=border_color,
                      interpolate=interpolate)
    
    if interactive:
        window.show(on_manager.scene)

    n_textures = bill.GetProperty().GetNumberOfTextures()
    texture = bill.GetProperty().GetTexture(tex_name)

    npt.assert_equal(1, n_textures)
    npt.assert_equal(WRAP_MODE_DIC[wrap_mode.lower()], texture.GetWrap())
    npt.assert_equal(BLENDING_MODE_DIC[blending_mode.lower()], texture.GetBlendingMode())
    npt.assert_array_almost_equal(list(border_color), texture.GetBorderColor())
    npt.assert_equal(interpolate, texture.GetInterpolate())
    npt.assert_equal(True, texture.GetMipmap())
    
def test_colormap_to_actor(interactive = False):

    fs_dec = import_fury_shader(os.path.join("effects", "color_mapping.glsl"))

    fs_impl = """
    vec2 res_factor = vec2(res.y/res.x, 1.0);
    vec2 renorm_tex = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
    float intensity = renorm_tex.x;
    vec4 tex = color_mapping(intensity, colormapTexture);
    fragOutput0 = vec4(tex);
    """

    on_scene = window.Scene()
    on_scene.set_camera(position=(-2, 1, -2),
                    focal_point=(0.0,
                                 0.0,
                                 0.0),
                    view_up=(0.0, 0.0, 0.0))

    on_manager = window.ShowManager(
        on_scene,
        "on_manager",
        (width,
        height))

    scale = 3.4*np.array([[width/height, 1.0, 0.0]])
    bill = billboard(np.array([[0.0, 0.0, 0.0]]), scales = scale, fs_impl=fs_impl, fs_dec=fs_dec)
    shader_custom_uniforms(bill, "fragment").SetUniform2f("res", on_manager.size)

    on_manager.scene.add(bill)

    tex_name = "colormapTexture"
    interpolate = True

    colormap_to_texture(create_colormap(np.arange(0.0, 1.0, 1/256), "viridis"), 
                        tex_name, 
                        bill, 
                        interpolate)

    
    if interactive:
        window.show(on_manager.scene)

    n_textures = bill.GetProperty().GetNumberOfTextures()
    texture = bill.GetProperty().GetTexture(tex_name)

    npt.assert_equal(1, n_textures)
    npt.assert_equal(WRAP_MODE_DIC["ClampToEdge".lower()], texture.GetWrap())
    npt.assert_equal(BLENDING_MODE_DIC["None".lower()], texture.GetBlendingMode())
    npt.assert_equal(interpolate, texture.GetInterpolate())
    npt.assert_equal(True, texture.GetMipmap())

def test_effect_manager_setup(interactive = False):

    scene = Scene()
    scene.set_camera(position=(-24, 20, -40),
                    focal_point=(0.0,
                                 0.0,
                                 0.0),
                    view_up=(0.0, 0.0, 0.0))

    manager = ShowManager(
        scene,
        "demo",
        (width,
        height))

    manager.initialize()

    em = EffectManager(manager)

    npt.assert_equal(True, em.scene.get_camera() == manager.scene.get_camera())
    npt.assert_equal(True, manager == em.on_manager)
    npt.assert_array_equal(manager.window.GetSize(), em.off_manager.window.GetSize())
    npt.assert_equal(True, em.scene == em.off_manager.scene)
    npt.assert_equal(True, em.off_manager.window.GetOffScreenRendering())
    npt.assert_equal(0, em._n_active_effects)
    npt.assert_equal({}, em._active_effects)

def test_remove_effect(interactive = False):
    scene = window.Scene()
    scene.set_camera(position=(-24, 20, -40),
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

    em = EffectManager(manager)

    kde_actor = em.kde(points, sigmas, colormap="inferno")

    manager.scene.add(kde_actor)
    em.remove_effect(kde_actor)

    if interactive:
        window.show(manager.scene)

    off_ascene = window.analyze_scene(em.off_manager.scene)
    on_ascene = window.analyze_scene(manager.scene)

    npt.assert_equal(0, em.on_manager.iren.HasObserver("RenderEvent"))
    npt.assert_equal(0, on_ascene.actors)
    npt.assert_equal(0, off_ascene.actors)
    npt.assert_equal({}, em._active_effects)
    npt.assert_equal(0, em._n_active_effects)


    