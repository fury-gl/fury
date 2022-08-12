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
from vtkmodules.vtkRenderingCore import (vtkLight as Light,
                                         vtkLightActor as LightActor)


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


def change_slice_position_x(slider):
    global light_params
    light_params['position'][0] = slider.value
    light.SetPosition(light_params['position'])


def change_slice_position_y(slider):
    global light_params
    light_params['position'][1] = slider.value
    light.SetPosition(light_params['position'])


def change_slice_position_z(slider):
    global light_params
    light_params['position'][2] = slider.value
    light.SetPosition(light_params['position'])


def change_slice_focal_point_x(slider):
    global light_params
    light_params['focal_point'][0] = slider.value
    light.SetFocalPoint(light_params['focal_point'])


def change_slice_focal_point_y(slider):
    global light_params
    light_params['focal_point'][1] = slider.value
    light.SetFocalPoint(light_params['focal_point'])


def change_slice_focal_point_z(slider):
    global light_params
    light_params['focal_point'][2] = slider.value
    light.SetFocalPoint(light_params['focal_point'])


def change_slice_cone_angle(slider):
    global light_params
    light_params['cone_angle'] = slider.value
    light.SetConeAngle(light_params['cone_angle'])


def change_slice_intensity(slider):
    global light_params
    light_params['intensity'] = slider.value
    light.SetIntensity(light_params['intensity'])


def change_slice_ambient_color_r(slider):
    global light_params
    light_params['ambient_color'][0] = slider.value
    light.SetAmbientColor(light_params['ambient_color'])


def change_slice_ambient_color_g(slider):
    global light_params
    light_params['ambient_color'][1] = slider.value
    light.SetAmbientColor(light_params['ambient_color'])


def change_slice_ambient_color_b(slider):
    global light_params
    light_params['ambient_color'][2] = slider.value
    light.SetAmbientColor(light_params['ambient_color'])


def change_slice_diffuse_color_r(slider):
    global light_params
    light_params['diffuse_color'][0] = slider.value
    light.SetDiffuseColor(light_params['diffuse_color'])


def change_slice_diffuse_color_g(slider):
    global light_params
    light_params['diffuse_color'][1] = slider.value
    light.SetDiffuseColor(light_params['diffuse_color'])


def change_slice_diffuse_color_b(slider):
    global light_params
    light_params['diffuse_color'][2] = slider.value
    light.SetDiffuseColor(light_params['diffuse_color'])


def change_slice_specular_color_r(slider):
    global light_params
    light_params['specular_color'][0] = slider.value
    light.SetSpecularColor(light_params['specular_color'])


def change_slice_specular_color_g(slider):
    global light_params
    light_params['specular_color'][1] = slider.value
    light.SetSpecularColor(light_params['specular_color'])


def change_slice_specular_color_b(slider):
    global light_params
    light_params['specular_color'][2] = slider.value
    light.SetSpecularColor(light_params['specular_color'])


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
    global light_panel, pbr_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        light_panel.re_align(size_change)
        pbr_panel.re_align(size_change)


if __name__ == '__main__':
    global doa, light, light_panel, light_params, obj_actor, pbr_panel, \
        pbr_params, size

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

    coords_min = np.min(bundle.get_data(), axis=0)
    coords_min_int = np.rint(coords_min).astype(int)
    coords_max = np.max(bundle.get_data(), axis=0)
    coords_max_int = np.rint(coords_max).astype(int)
    coords_abs_max = np.max(np.abs(np.stack((coords_min_int, coords_max_int),
                                            axis=0)), axis=0)
    coords_avg = np.rint((coords_min + coords_max) / 2).astype(int)

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

    light_params = {
        'position': [coords_avg[0], coords_max_int[1], 0],
        'focal_point': coords_avg, 'cone_angle': 80, 'intensity': .25,
        'ambient_color': [0, 0, 0], 'diffuse_color': [1, 1, 1],
        'specular_color': [0, 0, 0]}

    light = Light()
    light.SetLightTypeToSceneLight()
    #light.SetLightTypeToHeadlight()
    #light.SetLightTypeToCameraLight()
    light.SetPositional(True)
    light.SetPosition(light_params['position'])
    light.SetFocalPoint(light_params['focal_point'])
    light.SetConeAngle(light_params['cone_angle'])
    light.SetIntensity(light_params['intensity'])
    #light.SetColor(1, 1, 1)
    light.SetAmbientColor(light_params['ambient_color'])
    light.SetDiffuseColor(light_params['diffuse_color'])
    light.SetSpecularColor(light_params['specular_color'])

    scene.AddLight(light)

    light_actor = LightActor()
    light_actor.SetLight(light)
    #light_actor.GetFrustumProperty().SetColor(1, 1, 1)

    scene.add(light_actor)

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

    light_panel = ui.Panel2D(
        (380, 820), position=(5, 5), color=(.25, .25, .25), opacity=.75,
        align='right')

    panel_label_light = ui.TextBlock2D(text='Light', font_size=18, bold=True)
    slider_label_position_x = ui.TextBlock2D(text='Position X', font_size=16)
    slider_label_position_y = ui.TextBlock2D(text='Position Y', font_size=16)
    slider_label_position_z = ui.TextBlock2D(text='Position Z', font_size=16)
    slider_label_focal_point_x = ui.TextBlock2D(
        text='Focal Point X', font_size=16)
    slider_label_focal_point_y = ui.TextBlock2D(
        text='Focal Point Y', font_size=16)
    slider_label_focal_point_z = ui.TextBlock2D(
        text='Focal Point Z', font_size=16)
    slider_label_cone_angle = ui.TextBlock2D(text='Cone Angle', font_size=16)
    slider_label_intensity = ui.TextBlock2D(text='Intensity', font_size=16)
    slider_label_ambient_color_r = ui.TextBlock2D(
        text='Ambient Color R', font_size=16)
    slider_label_ambient_color_g = ui.TextBlock2D(
        text='Ambient Color G', font_size=16)
    slider_label_ambient_color_b = ui.TextBlock2D(
        text='Ambient Color B', font_size=16)
    slider_label_diffuse_color_r = ui.TextBlock2D(
        text='Diffuse Color R', font_size=16)
    slider_label_diffuse_color_g = ui.TextBlock2D(
        text='Diffuse Color G', font_size=16)
    slider_label_diffuse_color_b = ui.TextBlock2D(
        text='Diffuse Color B', font_size=16)
    slider_label_specular_color_r = ui.TextBlock2D(
        text='Specular Color R', font_size=16)
    slider_label_specular_color_g = ui.TextBlock2D(
        text='Specular Color G', font_size=16)
    slider_label_specular_color_b = ui.TextBlock2D(
        text='Specular Color B', font_size=16)

    label_pad_x = .04

    light_panel.add_element(panel_label_light, (.01, .97))
    light_panel.add_element(slider_label_position_x, (label_pad_x, .91))
    light_panel.add_element(slider_label_position_y, (label_pad_x, .86))
    light_panel.add_element(slider_label_position_z, (label_pad_x, .80))
    light_panel.add_element(slider_label_focal_point_x, (label_pad_x, .75))
    light_panel.add_element(slider_label_focal_point_y, (label_pad_x, .69))
    light_panel.add_element(slider_label_focal_point_z, (label_pad_x, .64))
    light_panel.add_element(slider_label_cone_angle, (label_pad_x, .58))
    light_panel.add_element(slider_label_intensity, (label_pad_x, .53))
    light_panel.add_element(slider_label_ambient_color_r, (label_pad_x, .47))
    light_panel.add_element(slider_label_ambient_color_g, (label_pad_x, .42))
    light_panel.add_element(slider_label_ambient_color_b, (label_pad_x, .36))
    light_panel.add_element(slider_label_diffuse_color_r, (label_pad_x, .31))
    light_panel.add_element(slider_label_diffuse_color_g, (label_pad_x, .25))
    light_panel.add_element(slider_label_diffuse_color_b, (label_pad_x, .20))
    light_panel.add_element(slider_label_specular_color_r, (label_pad_x, .14))
    light_panel.add_element(slider_label_specular_color_g, (label_pad_x, .09))
    light_panel.add_element(slider_label_specular_color_b, (label_pad_x, .03))

    slider_length = 200
    slider_slice_position_x = ui.LineSlider2D(
        initial_value=light_params['position'][0],
        min_value=-coords_abs_max[0], max_value=coords_abs_max[0],
        length=slider_length, text_template='{value:.0f}')
    slider_slice_position_y = ui.LineSlider2D(
        initial_value=light_params['position'][1],
        min_value=-coords_abs_max[1], max_value=coords_abs_max[1],
        length=slider_length, text_template='{value:.0f}')
    slider_slice_position_z = ui.LineSlider2D(
        initial_value=light_params['position'][2],
        min_value=-coords_abs_max[2], max_value=coords_abs_max[2],
        length=slider_length, text_template='{value:.0f}')

    slider_slice_focal_point_x = ui.LineSlider2D(
        initial_value=light_params['focal_point'][0],
        min_value=coords_min_int[0], max_value=coords_max_int[0],
        length=slider_length, text_template='{value:.0f}')
    slider_slice_focal_point_y = ui.LineSlider2D(
        initial_value=light_params['focal_point'][1],
        min_value=coords_min_int[1], max_value=coords_max_int[1],
        length=slider_length, text_template='{value:.0f}')
    slider_slice_focal_point_z = ui.LineSlider2D(
        initial_value=light_params['focal_point'][2],
        min_value=coords_min_int[2], max_value=coords_max_int[2],
        length=slider_length, text_template='{value:.0f}')

    slider_slice_cone_angle = ui.LineSlider2D(
        initial_value=light_params['cone_angle'], max_value=89,
        length=slider_length, text_template='{value:.0f}')

    slider_slice_intensity = ui.LineSlider2D(
        initial_value=light_params['intensity'], max_value=1,
        length=slider_length, text_template='{value:.1f}')

    slider_slice_ambient_color_r = ui.LineSlider2D(
        initial_value=light_params['ambient_color'][0], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_ambient_color_g = ui.LineSlider2D(
        initial_value=light_params['ambient_color'][1], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_ambient_color_b = ui.LineSlider2D(
        initial_value=light_params['ambient_color'][2], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_diffuse_color_r = ui.LineSlider2D(
        initial_value=light_params['diffuse_color'][0], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_diffuse_color_g = ui.LineSlider2D(
        initial_value=light_params['diffuse_color'][1], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_diffuse_color_b = ui.LineSlider2D(
        initial_value=light_params['diffuse_color'][2], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_specular_color_r = ui.LineSlider2D(
        initial_value=light_params['specular_color'][0], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_specular_color_g = ui.LineSlider2D(
        initial_value=light_params['specular_color'][1], max_value=1,
        length=slider_length, text_template='{value:.1f}')
    slider_slice_specular_color_b = ui.LineSlider2D(
        initial_value=light_params['specular_color'][2], max_value=1,
        length=slider_length, text_template='{value:.1f}')

    slider_slice_position_x.on_change = change_slice_position_x
    slider_slice_position_y.on_change = change_slice_position_y
    slider_slice_position_z.on_change = change_slice_position_z
    slider_slice_focal_point_x.on_change = change_slice_focal_point_x
    slider_slice_focal_point_y.on_change = change_slice_focal_point_y
    slider_slice_focal_point_z.on_change = change_slice_focal_point_z
    slider_slice_cone_angle.on_change = change_slice_cone_angle
    slider_slice_intensity.on_change = change_slice_intensity
    slider_slice_ambient_color_r.on_change = change_slice_ambient_color_r
    slider_slice_ambient_color_g.on_change = change_slice_ambient_color_g
    slider_slice_ambient_color_b.on_change = change_slice_ambient_color_b
    slider_slice_diffuse_color_r.on_change = change_slice_diffuse_color_r
    slider_slice_diffuse_color_g.on_change = change_slice_diffuse_color_g
    slider_slice_diffuse_color_b.on_change = change_slice_diffuse_color_b
    slider_slice_specular_color_r.on_change = change_slice_specular_color_r
    slider_slice_specular_color_g.on_change = change_slice_specular_color_g
    slider_slice_specular_color_b.on_change = change_slice_specular_color_b

    light_slice_pad_x = .4
    light_panel.add_element(slider_slice_position_x, (light_slice_pad_x, .91))
    light_panel.add_element(slider_slice_position_y, (light_slice_pad_x, .86))
    light_panel.add_element(slider_slice_position_z, (light_slice_pad_x, .80))
    light_panel.add_element(
        slider_slice_focal_point_x, (light_slice_pad_x, .75))
    light_panel.add_element(
        slider_slice_focal_point_y, (light_slice_pad_x, .69))
    light_panel.add_element(
        slider_slice_focal_point_z, (light_slice_pad_x, .64))
    light_panel.add_element(slider_slice_cone_angle, (light_slice_pad_x, .58))
    light_panel.add_element(slider_slice_intensity, (light_slice_pad_x, .53))
    light_panel.add_element(
        slider_slice_ambient_color_r, (light_slice_pad_x, .47))
    light_panel.add_element(
        slider_slice_ambient_color_g, (light_slice_pad_x, .42))
    light_panel.add_element(
        slider_slice_ambient_color_b, (light_slice_pad_x, .36))
    light_panel.add_element(
        slider_slice_diffuse_color_r, (light_slice_pad_x, .31))
    light_panel.add_element(
        slider_slice_diffuse_color_g, (light_slice_pad_x, .25))
    light_panel.add_element(
        slider_slice_diffuse_color_b, (light_slice_pad_x, .20))
    light_panel.add_element(
        slider_slice_specular_color_r, (light_slice_pad_x, .14))
    light_panel.add_element(
        slider_slice_specular_color_g, (light_slice_pad_x, .09))
    light_panel.add_element(
        slider_slice_specular_color_b, (light_slice_pad_x, .03))

    scene.add(light_panel)

    pbr_panel = ui.Panel2D(
        (420, 500), position=(1495, 5), color=(.25, .25, .25), opacity=.75,
        align='right')

    panel_label_pbr = ui.TextBlock2D(text='PBR', font_size=18, bold=True)
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

    pbr_panel.add_element(panel_label_pbr, (.01, .95))
    pbr_panel.add_element(slider_label_metallic, (label_pad_x, .87))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .79))
    pbr_panel.add_element(slider_label_anisotropy, (label_pad_x, .70))
    pbr_panel.add_element(slider_label_anisotropy_rotation, (label_pad_x, .62))
    pbr_panel.add_element(
        slider_label_anisotropy_direction_x, (label_pad_x, .54))
    pbr_panel.add_element(
        slider_label_anisotropy_direction_y, (label_pad_x, .46))
    pbr_panel.add_element(
        slider_label_anisotropy_direction_z, (label_pad_x, .38))
    pbr_panel.add_element(slider_label_coat_strength, (label_pad_x, .30))
    pbr_panel.add_element(slider_label_coat_roughness, (label_pad_x, .21))
    pbr_panel.add_element(slider_label_base_ior, (label_pad_x, .13))
    pbr_panel.add_element(slider_label_coat_ior, (label_pad_x, .05))

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

    slider_slice_anisotropy_direction_x = ui.LineSlider2D(
        initial_value=doa[0], min_value=-1, max_value=1, length=slider_length,
        text_template='{value:.1f}')
    slider_slice_anisotropy_direction_y = ui.LineSlider2D(
        initial_value=doa[1], min_value=-1, max_value=1, length=slider_length,
        text_template='{value:.1f}')
    slider_slice_anisotropy_direction_z = ui.LineSlider2D(
        initial_value=doa[2], min_value=-1, max_value=1, length=slider_length,
        text_template='{value:.1f}')

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

    pbr_slice_pad_x = .46
    pbr_panel.add_element(slider_slice_metallic, (pbr_slice_pad_x, .87))
    pbr_panel.add_element(slider_slice_roughness, (pbr_slice_pad_x, .79))
    pbr_panel.add_element(slider_slice_anisotropy, (pbr_slice_pad_x, .70))
    pbr_panel.add_element(
        slider_slice_anisotropy_rotation, (pbr_slice_pad_x, .62))
    pbr_panel.add_element(
        slider_slice_anisotropy_direction_x, (pbr_slice_pad_x, .54))
    pbr_panel.add_element(
        slider_slice_anisotropy_direction_y, (pbr_slice_pad_x, .46))
    pbr_panel.add_element(
        slider_slice_anisotropy_direction_z, (pbr_slice_pad_x, .38))
    pbr_panel.add_element(slider_slice_coat_strength, (pbr_slice_pad_x, .30))
    pbr_panel.add_element(slider_slice_coat_roughness, (pbr_slice_pad_x, .21))
    pbr_panel.add_element(slider_slice_base_ior, (pbr_slice_pad_x, .13))
    pbr_panel.add_element(slider_slice_coat_ior, (pbr_slice_pad_x, .05))

    scene.add(pbr_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
