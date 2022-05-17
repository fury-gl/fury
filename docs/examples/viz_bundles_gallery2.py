import numpy as np
import os

from dipy.io.streamline import load_tractogram
from dipy.data.fetcher import get_bundle_atlas_hcp842
from dipy.stats.analysis import assignment_map
from fury import actor, ui, window
from fury.colormap import line_colors
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture, load_image
from fury.lib import ImageData, Texture, numpy_support
from fury.material import manifest_pbr
from fury.shaders import shader_to_actor
from fury.utils import (normals_from_actor, numpy_to_vtk_colors, rotate,
                        tangents_from_direction_of_anisotropy,
                        tangents_to_actor, update_polydata_normals)
from vtkmodules.vtkRenderingCore import vtkLight as Light


def change_slice_metallic(slider):
    global pbr_params
    pbr_params.metallic = slider.value


def change_slice_roughness(slider):
    global pbr_params
    pbr_params.roughness = slider.value


def change_slice_anisotropy(slider):
    global pbr_params
    pbr_params.anisotropy = slider.value


def change_slice_anisotropy_direction_x(slider):
    global doa, normals, obj_actor
    doa[0] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(obj_actor, tangents)


def change_slice_anisotropy_direction_y(slider):
    global doa, normals, obj_actor
    doa[1] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(obj_actor, tangents)


def change_slice_anisotropy_direction_z(slider):
    global doa, normals, obj_actor
    doa[2] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(obj_actor, tangents)


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


def get_cubemap_from_ndarrays(array, flip=True):
    texture = Texture()
    texture.CubeMapOn()
    for idx, img in enumerate(array):
        vtk_img = ImageData()
        vtk_img.SetDimensions(img.shape[1], img.shape[0], 1)
        if flip:
            # Flip horizontally
            vtk_arr = numpy_support.numpy_to_vtk(np.flip(
                img.swapaxes(0, 1), axis=1).reshape((-1, 3), order='F'))
        else:
            vtk_arr = numpy_support.numpy_to_vtk(img.reshape((-1, 3),
                                                             order='F'))
        vtk_arr.SetName('Image')
        vtk_img.GetPointData().AddArray(vtk_arr)
        vtk_img.GetPointData().SetActiveScalars('Image')
        texture.SetInputDataObject(idx, vtk_img)
    return texture


def read_texture(fname):
    if os.path.isfile(fname):
        img = load_image(fname)
        vtk_img_arr = numpy_support.numpy_to_vtk(img.reshape((-1, 3),
                                                             order='F'))
        vtk_img_arr.SetName('Image')
        vtk_img = ImageData()
        vtk_img.GetPointData().AddArray(vtk_img_arr)
        vtk_img.GetPointData().SetActiveScalars('Image')
        texture = Texture()
        texture.SetInputDataObject(vtk_img)
        return texture


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        control_panel.re_align(size_change)


if __name__ == '__main__':
    global control_panel, doa, obj_actor, control_panel, pbr_params, size

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
    stats_dir = '/run/media/guaje/Data/Downloads/buan_flow/lmm_plots'

    tractograms = ['AF_L.trk', 'AF_R.trk', 'CST_L.trk', 'CST_R.trk']
    stats = ['AF_L_fa_pvalues.npy', 'AF_R_fa_pvalues.npy',
             'CST_L_fa_pvalues.npy', 'CST_R_fa_pvalues.npy']
    buan_highlights = [(1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0)]
    num_cols = len(tractograms)

    doa = [0, 1, .5]
    buan_thr = .05

    # Load tractogram
    tract_file = os.path.join(bundles_dir, tractograms[0])
    sft = load_tractogram(tract_file, 'same', bbox_valid_check=False)
    bundle = sft.streamlines

    num_lines = len(bundle)
    lines_range = range(num_lines)
    points_per_line = [len(bundle[i]) for i in lines_range]
    points_per_line = np.array(points_per_line, np.intp)

    cols_arr = line_colors(bundle)
    colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
    vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
    colors = numpy_support.vtk_to_numpy(vtk_colors)
    colors = (colors - np.min(colors)) / np.ptp(colors)

    # Load stats
    stat_file = os.path.join(stats_dir, stats[0])
    p_values = np.load(stat_file)

    data_length = len(p_values)

    indx = assignment_map(bundle, bundle, data_length)
    ind = np.array(indx)

    for j in range(data_length):
        if p_values[j] < buan_thr:
            colors[ind == j] = buan_highlights[0]

    obj_actor = actor.streamtube(bundle, colors=colors, linewidth=.25)
    normals = normals_from_actor(obj_actor)
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(obj_actor, tangents)

    # AF_L rotation
    rotate(obj_actor, rotation=(-90, 1, 0, 0))
    rotate(obj_actor, rotation=(90, 0, 1, 0))

    # Actor rotation for brudslojan texture
    #rotate(obj_actor, rotation=(-110, 0, 1, 0))

    pbr_params = manifest_pbr(obj_actor, metallic=.25, anisotropy=1)

    scene.add(obj_actor)

    light = Light()
    light.SetLightTypeToSceneLight()
    light.SetPositional(True)
    light.SetPosition(-35, 5, 0)
    #light.SetConeAngle(10)
    light.SetFocalPoint(-35, -5, 0)
    #light.SetColor(0, 1, 0)
    light.SetDiffuseColor(1, 0, 0)
    light.SetAmbientColor(0, 1, 0)
    light.SetSpecularColor(0, 0, 1)
    light.SetIntensity(.75)
    scene.AddLight(light)

    #light_pink = Light()
    #light_pink.SetPositional(True)
    #light_pink.SetPosition(4, 5, 1)
    #light_pink.SetColor(1, 0, 1)
    #light_pink.SetIntensity(.6)
    #scene.AddLight(light_pink)

    emissive_color = (1, 0, 0)
    emissive_sphere = actor.sphere(np.array([[-35, -5, 0]]), emissive_color,
                                   radii=5)

    emissive_sphere.GetProperty().SetInterpolationToPBR()
    emissive_sphere.GetProperty().SetEmissiveFactor(emissive_color)
    emissive_pd = emissive_sphere.GetMapper().GetInput()
    update_polydata_normals(emissive_pd)
    fs_impl = \
    """
    emissiveColor = vec3(1);
    emissiveColor = emissiveColor * emissiveFactorUniform;
    color = iblDiffuse + iblSpecular;
    color += Lo;
    color = mix(color, color * ao, aoStrengthUniform);
    color += emissiveColor;
    color = pow(color, vec3(1.0/2.2));
    fragOutput0 = vec4(color, opacity);
    """
    shader_to_actor(emissive_sphere, 'fragment', impl_code=fs_impl,
                    block='light')#,
                    #debug=True)

    scene.add(emissive_sphere)

    scene.reset_camera()
    #scene.zoom(1.9)  # Glyptotek's zoom

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    control_panel = ui.Panel2D(
        (400, 500), position=(5, 5), color=(.25, .25, .25), opacity=.75,
        align='right')

    slider_label_metallic = ui.TextBlock2D(text='Metallic', font_size=16)
    slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)
    slider_label_anisotropy = ui.TextBlock2D(text='Anisotropy', font_size=16)
    slider_label_anisotropy_rotation = ui.TextBlock2D(
        text='Anisotropy Rotation', font_size=16)
    slider_label_anisotropy_direction_x = ui.TextBlock2D(
        text='Anisotropy Direction X', font_size=16)
    slider_label_anisotropy_direction_y = ui.TextBlock2D(
        text='Anisotropy Direction Y', font_size=16)
    slider_label_anisotropy_direction_z = ui.TextBlock2D(
        text='Anisotropy Direction Z', font_size=16)
    slider_label_coat_strength = ui.TextBlock2D(text='Coat Strength',
                                                font_size=16)
    slider_label_coat_roughness = ui.TextBlock2D(
        text='Coat Roughness', font_size=16)
    slider_label_base_ior = ui.TextBlock2D(text='Base IoR', font_size=16)
    slider_label_coat_ior = ui.TextBlock2D(text='Coat IoR', font_size=16)

    control_panel.add_element(slider_label_metallic, (.01, .95))
    control_panel.add_element(slider_label_roughness, (.01, .86))
    control_panel.add_element(slider_label_anisotropy, (.01, .77))
    control_panel.add_element(slider_label_anisotropy_rotation, (.01, .68))
    control_panel.add_element(slider_label_anisotropy_direction_x, (.01, .59))
    control_panel.add_element(slider_label_anisotropy_direction_y, (.01, .5))
    control_panel.add_element(slider_label_anisotropy_direction_z, (.01, .41))
    control_panel.add_element(slider_label_coat_strength, (.01, .32))
    control_panel.add_element(slider_label_coat_roughness, (.01, .23))
    control_panel.add_element(slider_label_base_ior, (.01, .14))
    control_panel.add_element(slider_label_coat_ior, (.01, .05))

    slider_slice_metallic = ui.LineSlider2D(
        initial_value=pbr_params.metallic, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=pbr_params.roughness, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_anisotropy = ui.LineSlider2D(
        initial_value=pbr_params.anisotropy, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_anisotropy_rotation = ui.LineSlider2D(
        initial_value=pbr_params.anisotropy_rotation, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_coat_strength = ui.LineSlider2D(
        initial_value=pbr_params.coat_strength, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_coat_roughness = ui.LineSlider2D(
        initial_value=pbr_params.coat_roughness, max_value=1, length=195,
        text_template='{value:.1f}')

    slider_slice_anisotropy_direction_x = ui.LineSlider2D(
        initial_value=doa[0], min_value=-1, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_anisotropy_direction_y = ui.LineSlider2D(
        initial_value=doa[1], min_value=-1, max_value=1, length=195,
        text_template='{value:.1f}')
    slider_slice_anisotropy_direction_z = ui.LineSlider2D(
        initial_value=doa[2], min_value=-1, max_value=1, length=195,
        text_template='{value:.1f}')

    slider_slice_base_ior = ui.LineSlider2D(
        initial_value=pbr_params.base_ior, min_value=1, max_value=2.3,
        length=195, text_template='{value:.02f}')
    slider_slice_coat_ior = ui.LineSlider2D(
        initial_value=pbr_params.coat_ior, min_value=1, max_value=2.3,
        length=195, text_template='{value:.02f}')

    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_roughness.on_change = change_slice_roughness
    slider_slice_anisotropy.on_change = change_slice_anisotropy
    slider_slice_anisotropy_rotation.on_change = change_slice_anisotropy_rotation
    slider_slice_anisotropy_direction_x.on_change = (
        change_slice_anisotropy_direction_x)
    slider_slice_anisotropy_direction_y.on_change = (
        change_slice_anisotropy_direction_y)
    slider_slice_anisotropy_direction_z.on_change = (
        change_slice_anisotropy_direction_z)
    slider_slice_coat_strength.on_change = change_slice_coat_strength
    slider_slice_coat_roughness.on_change = change_slice_coat_roughness
    slider_slice_base_ior.on_change = change_slice_base_ior
    slider_slice_coat_ior.on_change = change_slice_coat_ior

    control_panel.add_element(slider_slice_metallic, (.44, .95))
    control_panel.add_element(slider_slice_roughness, (.44, .86))
    control_panel.add_element(slider_slice_anisotropy, (.44, .77))
    control_panel.add_element(slider_slice_anisotropy_rotation, (.44, .68))
    control_panel.add_element(slider_slice_anisotropy_direction_x, (.44, .59))
    control_panel.add_element(slider_slice_anisotropy_direction_y, (.44, .5))
    control_panel.add_element(slider_slice_anisotropy_direction_z, (.44, .41))
    control_panel.add_element(slider_slice_coat_strength, (.44, .32))
    control_panel.add_element(slider_slice_coat_roughness, (.44, .23))
    control_panel.add_element(slider_slice_base_ior, (.44, .14))
    control_panel.add_element(slider_slice_coat_ior, (.44, .05))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
