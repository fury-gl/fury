import numpy as np
import os

from dipy.io.streamline import load_tractogram
from dipy.data.fetcher import get_bundle_atlas_hcp842
from fury import actor, ui, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.material import manifest_pbr
from fury.shaders import attribute_to_actor, shader_to_actor
from fury.utils import (normals_from_actor,
                        tangents_from_direction_of_anisotropy,
                        tangents_to_actor, vertices_from_actor)


def bundle_tangents(bundle):
    tangents = []
    for line in bundle:
        line_length = len(line)
        for i in range(line_length - 1):
            dif = line[i + 1] - line[i]
            dist = np.sqrt(np.sum(dif ** 2))
            tangents.append(dif / dist)
        tangents.append(dif/dist)
    tangents = np.array(tangents)
    return tangents


def change_slice_metallic(slider):
    global pbr_params
    pbr_params.metallic = slider.value


def change_slice_roughness(slider):
    global pbr_params
    pbr_params.roughness = slider.value


def change_slice_anisotropy(slider):
    global pbr_params
    pbr_params.anisotropy = slider.value


def change_slice_anisotropy_rotation(slider):
    global pbr_params
    pbr_params.anisotropy_rotation = slider.value


def change_slice_coat_strength(slider):
    global pbr_params
    pbr_params.coat_strength = slider.value


def change_slice_coat_roughness(slider):
    global pbr_params
    pbr_params.coat_roughness = slider.value


def change_slice_base_ior(slider):
    global pbr_params
    pbr_params.base_ior = slider.value


def change_slice_coat_ior(slider):
    global pbr_params
    pbr_params.coat_ior = slider.value


def win_callback(obj, event):
    global pbr_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        pbr_panel.re_align(size_change)


if __name__ == '__main__':
    global obj_actor, pbr_panel, pbr_params, size

    fetch_viz_cubemaps()

    #texture_name = 'skybox'
    texture_name = 'brudslojan'
    textures = read_viz_cubemap(texture_name)

    cubemap = load_cubemap_texture(textures)

    """
    img_shape = (1024, 1024)

    # Flip horizontally
    img_grad = np.flip(np.tile(np.linspace(0, 255, num=img_shape[0]),
                               (img_shape[1], 1)).astype(np.uint8), axis=1)
    cubemap_side_img = np.stack((img_grad,) * 3, axis=-1)

    cubemap_top_img = np.ones((img_shape[0], img_shape[1], 3)).astype(
        np.uint8) * 255

    cubemap_bottom_img = np.zeros((img_shape[0], img_shape[1], 3)).astype(
        np.uint8)

    cubemap_imgs = [cubemap_side_img, cubemap_side_img, cubemap_top_img,
                    cubemap_bottom_img, cubemap_side_img, cubemap_side_img]

    cubemap = get_cubemap_from_ndarrays(cubemap_imgs, flip=False)
    """

    #cubemap.RepeatOff()
    #cubemap.EdgeClampOn()

    scene = window.Scene()

    #scene = window.Scene(skybox=cubemap)
    #scene.skybox(gamma_correct=False)

    #scene.background((1, 1, 1))

    # Scene rotation for brudslojan texture
    #scene.yaw(-110)

    atlas, bundles = get_bundle_atlas_hcp842()
    bundles_dir = os.path.dirname(bundles)
    tractograms = ['AC.trk', 'CC_ForcepsMajor.trk', 'CC_ForcepsMinor.trk',
                   'CCMid.trk', 'F_L_R.trk', 'MCP.trk', 'PC.trk', 'SCP.trk',
                   'V.trk']

    # Load tractogram
    tract_file = os.path.join(bundles_dir, tractograms[4])
    sft = load_tractogram(tract_file, 'same', bbox_valid_check=False)
    bundle = sft.streamlines

    obj_actor = actor.line(bundle, lod=False)

    """
    tmp_line_idx = 107  # Shortest line
    #tmp_line_idx = 146  # Longest line
    tmp_line = bundle[tmp_line_idx]
    
    obj_actor = actor.line([tmp_line], lod=False)

    line_length = len(tmp_line)

    tangents = np.empty((line_length, 3))
    for i in range(line_length - 1):
        dif = tmp_line[i + 1] - tmp_line[i]
        dist = np.sqrt(np.sum(dif ** 2))
        tangents[i, :] = dif / dist
    tangents[line_length - 1, :] = tangents[line_length - 2, :]
    
    vertices = tmp_line
    """

    tangents = bundle_tangents(bundle)

    #vertices = vertices_from_actor(obj_actor)

    """
    tangent_len = .1
    tangents_endpnts = vertices + tangents * tangent_len

    # View tangents as dots
    #tangent_actor = actor.dot(tangents_endpnts, colors=(1, 0, 0))
    #scene.add(tangent_actor)

    # View tangents as lines
    tangent_lines = [[vertices[i, :], tangents_endpnts[i, :]] for i in
                     range(len(vertices))]
    tangent_actor = actor.line(tangent_lines, colors=(1, 0, 0))
    scene.add(tangent_actor)
    """

    tangents_to_actor(obj_actor, tangents)

    pbr_params = manifest_pbr(obj_actor, metallic=.25, anisotropy=1)

    #attribute_to_actor(obj_actor, tangents, 'tangent')

    vs_dec_clip = \
    """
    uniform mat4 MCVCMatrix;
    """
    #shader_to_actor(obj_actor, 'vertex', block='clip', decl_code=vs_dec_clip)

    vs_dec_vp = \
    """
    in vec3 tangent;
    
    out vec3 tangentVSOutput;
    
    float square(float x)
    {
        return x * x;
    }
    """
    vs_impl_vp = \
    """ 
    tangentVSOutput = tangent;
    """
    #shader_to_actor(obj_actor, 'vertex', block='valuepass',
    #                decl_code=vs_dec_vp, impl_code=vs_impl_vp)

    vs_impl_light = \
    """
    vec4 camPos = -MCVCMatrix[3] * MCVCMatrix;
    vec3 lightDir = normalize(vertexMC.xyz - camPos.xyz);
    float dotLN = sqrt(1 - square(dot(lightDir, tangent))); 
    """
    #shader_to_actor(obj_actor, 'vertex', block='light',
    #                impl_code=vs_impl_light)#, debug=True)
    #shader_to_actor(obj_actor, 'fragment', block='valuepass', debug=True)

    scene.add(obj_actor)

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    pbr_panel = ui.Panel2D(
        (420, 400), position=(1495, 5), color=(.25, .25, .25), opacity=.75,
        align='right')

    panel_label_pbr = ui.TextBlock2D(text='PBR', font_size=18, bold=True)
    slider_label_metallic = ui.TextBlock2D(text='Metallic', font_size=16)
    slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)
    slider_label_anisotropy = ui.TextBlock2D(text='Anisotropy', font_size=16)
    slider_label_anisotropy_rotation = ui.TextBlock2D(
        text='Anisotropy Rotation', font_size=16)
    slider_label_coat_strength = ui.TextBlock2D(text='Coat Strength',
                                                font_size=16)
    slider_label_coat_roughness = ui.TextBlock2D(
        text='Coat Roughness', font_size=16)
    slider_label_base_ior = ui.TextBlock2D(text='Base IoR', font_size=16)
    slider_label_coat_ior = ui.TextBlock2D(text='Coat IoR', font_size=16)

    label_pad_x = .04

    pbr_panel.add_element(panel_label_pbr, (.01, .925))
    pbr_panel.add_element(slider_label_metallic, (label_pad_x, .819))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .713))
    pbr_panel.add_element(slider_label_anisotropy, (label_pad_x, .606))
    pbr_panel.add_element(slider_label_anisotropy_rotation, (label_pad_x, .5))
    pbr_panel.add_element(slider_label_coat_strength, (label_pad_x, .394))
    pbr_panel.add_element(slider_label_coat_roughness, (label_pad_x, .288))
    pbr_panel.add_element(slider_label_base_ior, (label_pad_x, .181))
    pbr_panel.add_element(slider_label_coat_ior, (label_pad_x, .075))

    slider_length = 200

    slider_slice_metallic = ui.LineSlider2D(
        initial_value=pbr_params.metallic, max_value=1, length=slider_length,
        text_template='{value:.1f}')
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=pbr_params.roughness, max_value=1, length=slider_length,
        text_template='{value:.1f}')
    slider_slice_anisotropy = ui.LineSlider2D(
        initial_value=pbr_params.anisotropy, max_value=1, length=slider_length,
        text_template='{value:.1f}')
    slider_slice_anisotropy_rotation = ui.LineSlider2D(
        initial_value=pbr_params.anisotropy_rotation, max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_coat_strength = ui.LineSlider2D(
        initial_value=pbr_params.coat_strength, max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_coat_roughness = ui.LineSlider2D(
        initial_value=pbr_params.coat_roughness, max_value=1,
        length=slider_length, text_template='{value:.1f}')

    slider_slice_base_ior = ui.LineSlider2D(
        initial_value=pbr_params.base_ior, min_value=1, max_value=2.3,
        length=slider_length, text_template='{value:.02f}')
    slider_slice_coat_ior = ui.LineSlider2D(
        initial_value=pbr_params.coat_ior, min_value=1, max_value=2.3,
        length=slider_length, text_template='{value:.02f}')

    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_roughness.on_change = change_slice_roughness
    slider_slice_anisotropy.on_change = change_slice_anisotropy
    slider_slice_anisotropy_rotation.on_change = change_slice_anisotropy_rotation
    slider_slice_coat_strength.on_change = change_slice_coat_strength
    slider_slice_coat_roughness.on_change = change_slice_coat_roughness
    slider_slice_base_ior.on_change = change_slice_base_ior
    slider_slice_coat_ior.on_change = change_slice_coat_ior

    pbr_slice_pad_x = .46
    pbr_panel.add_element(slider_slice_metallic, (pbr_slice_pad_x, .819))
    pbr_panel.add_element(slider_slice_roughness, (pbr_slice_pad_x, .713))
    pbr_panel.add_element(slider_slice_anisotropy, (pbr_slice_pad_x, .606))
    pbr_panel.add_element(
        slider_slice_anisotropy_rotation, (pbr_slice_pad_x, .5))
    pbr_panel.add_element(slider_slice_coat_strength, (pbr_slice_pad_x, .394))
    pbr_panel.add_element(slider_slice_coat_roughness, (pbr_slice_pad_x, .288))
    pbr_panel.add_element(slider_slice_base_ior, (pbr_slice_pad_x, .181))
    pbr_panel.add_element(slider_slice_coat_ior, (pbr_slice_pad_x, .075))

    scene.add(pbr_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
