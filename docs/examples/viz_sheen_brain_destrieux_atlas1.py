from datetime import timedelta
from fury import actor, ui, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.lib import ImageData, PolyData, Texture, numpy_support
from fury.material import manifest_principled
from fury.utils import (get_actor_from_polydata, get_polydata_normals,
                        normals_from_v_f, set_polydata_colors,
                        set_polydata_normals, set_polydata_triangles,
                        set_polydata_vertices, update_polydata_normals)
from matplotlib import cm
from nibabel import gifti
from nilearn import datasets, surface
from time import time


import gzip
import numpy as np


def change_slice_subsurface(slider):
    global principled_params
    principled_params['subsurface'] = slider.value


def change_slice_metallic(slider):
    global left_hemi_actor, principled_params
    principled_params['metallic'] = slider.value
    left_hemi_actor.GetProperty().SetMetallic(slider.value)


def change_slice_specular(slider):
    global left_hemi_actor, principled_params
    principled_params['specular'] = slider.value
    left_hemi_actor.GetProperty().SetSpecular(slider.value)


def change_slice_specular_tint(slider):
    global principled_params
    principled_params['specular_tint'] = slider.value


def change_slice_roughness(slider):
    global left_hemi_actor, principled_params
    principled_params['roughness'] = slider.value
    left_hemi_actor.GetProperty().SetRoughness(slider.value)


def change_slice_anisotropic(slider):
    global principled_params
    principled_params['anisotropic'] = slider.value


def change_slice_sheen(slider):
    global principled_params
    principled_params['sheen'] = slider.value


def change_slice_sheen_tint(slider):
    global principled_params
    principled_params['sheen_tint'] = slider.value


def change_slice_clearcoat(slider):
    global principled_params
    principled_params['clearcoat'] = slider.value


def change_slice_clearcoat_gloss(slider):
    global principled_params
    principled_params['clearcoat_gloss'] = slider.value


def change_slice_subsurf_r(slider):
    global principled_params
    principled_params['subsurface_color'][0] = slider.value


def change_slice_subsurf_g(slider):
    global principled_params
    principled_params['subsurface_color'][1] = slider.value


def change_slice_subsurf_b(slider):
    global principled_params
    principled_params['subsurface_color'][2] = slider.value


def change_slice_aniso_x(slider):
    global principled_params
    principled_params['anisotropic_direction'][0] = slider.value


def change_slice_aniso_y(slider):
    global principled_params
    principled_params['anisotropic_direction'][1] = slider.value


def change_slice_aniso_z(slider):
    global principled_params
    principled_params['anisotropic_direction'][2] = slider.value


def change_slice_opacity(slider):
    global left_hemi_actor
    left_hemi_actor.GetProperty().SetOpacity(slider.value)


def compute_texture_colors(textures, max_val, min_val=None, cmap='gist_ncar'):
    color_cmap = cm.get_cmap(cmap)
    textures_shape = textures.shape
    # TODO: Evaluate move
    colors = np.empty(textures_shape + (3,))
    if textures_shape[1] == 1:
        print('Computing colors from texture...')
        if min_val is not None:
            vals = (textures - min_val) / (max_val - min_val)
            colors = np.array([color_cmap(v)[0, :3] for v in vals])
        else:
            # TODO: Replace with numpy version
            for i in range(textures_shape[0]):
                # Normalize values between [0, 1]
                val = (textures[i] + max_val) / (2 * max_val)
                colors[i] = np.array(color_cmap(val))[0, :3]
    else:
        for j in range(textures_shape[1]):
            print('Computing colors for texture {:02d}/{}'.format(
                j + 1, textures_shape[1]))
            # TODO: Replace with numpy version
            for i in range(textures_shape[0]):
                # Normalize values between [0, 1]
                val = (textures[i, j] + max_val) / (2 * max_val)
                colors[i, j] = np.array(color_cmap(val))[:3]
    colors *= 255
    return colors


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


def get_hemisphere_actor(fname, colors=None, auto_normals='vtk'):
    points, triangles = surface.load_surf_mesh(fname)
    polydata = PolyData()
    set_polydata_vertices(polydata, points)
    set_polydata_triangles(polydata, triangles)
    if auto_normals.lower() == 'vtk':
        update_polydata_normals(polydata)
    elif auto_normals.lower() == 'fury':
        normals = normals_from_v_f(points, triangles)
        set_polydata_normals(polydata, normals)
    if colors is not None:
        if type(colors) == str:
            if colors.lower() == 'normals':
                if auto_normals.lower() == 'vtk':
                    normals = get_polydata_normals(polydata)
                if normals is not None:
                    colors = (normals + 1) / 2 * 255
        set_polydata_colors(polydata, colors)
    return get_actor_from_polydata(polydata)


def points_from_gzipped_gifti(fname):
    with gzip.open(fname) as f:
        as_bytes = f.read()
    parser = gifti.GiftiImage.parser()
    parser.parse(as_bytes)
    gifti_img = parser.img
    return gifti_img.darrays[0].data


def win_callback(obj, event):
    global control_panel, params_panel, principled_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        params_panel.re_align(size_change)
        principled_panel.re_align(size_change)
        control_panel.re_align(size_change)


if __name__ == '__main__':
    global control_panel, left_hemi_actor, params_panel, principled_panel, \
        principled_params, size

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

    scene.background((1, 1, 1))

    # Scene rotation for brudslojan texture
    #scene.yaw(-110)

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

    left_parcellation = destrieux_atlas.map_left
    right_parcellation = destrieux_atlas.map_right

    fsaverage = datasets.fetch_surf_fsaverage()

    left_pial_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    left_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_left)

    right_pial_mesh = surface.load_surf_mesh(fsaverage.pial_right)
    right_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_right)

    """
    from nilearn.plotting import plot_surf_roi
    import matplotlib.pyplot as plt

    plot_surf_roi(fsaverage.pial_left, left_parcellation,
                  bg_map=fsaverage.sulc_left, bg_on_data=True, darkness=.5)
    plt.show()
    """

    labels = destrieux_atlas.labels

    min_val = np.nanmin(left_parcellation)
    max_val = 1 + np.nanmax(left_parcellation)

    left_coordinates = []
    right_coordinates = []
    connectome_colors = []
    cmap = cm.get_cmap('gist_ncar')
    t = time()
    for i, label in enumerate(labels):
        if 'Unknown' not in str(label):  # Omit the Unknown label.
            # Compute mean location of vertices in label of index k
            left_coordinates.append(np.mean(left_pial_mesh.coordinates[
                left_parcellation == i, :], axis=0))
            right_coordinates.append(np.mean(right_pial_mesh.coordinates[
                right_parcellation == i, :], axis=0))
            connectome_colors.append(
                cmap((i - min_val) / (max_val - min_val))[:3])
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    # 3D coordinates of parcels
    left_coordinates = np.array(left_coordinates)
    right_coordinates = np.array(right_coordinates)

    # Connectome colors
    connectome_colors = np.array(connectome_colors) * 255

    left_nodes_actor = actor.sphere(left_coordinates, (1, 0, 0), opacity=.25)
    right_nodes_actor = actor.sphere(right_coordinates, (1, 0, 0), opacity=.25)
    
    scene.add(left_nodes_actor)
    scene.add(right_nodes_actor)

    edges = [[left_coordinates[i], right_coordinates[i]] for i in range(
            len(left_coordinates))]
    edges_actor = actor.streamtube(edges, (1, 0, 0), linewidth=.5,
                                   opacity=.25)

    scene.add(edges_actor)

    left_max_op_vals = -np.nanmin(left_sulc_points)
    left_min_op_vals = -np.nanmax(left_sulc_points)

    left_opacities = ((-left_sulc_points - left_min_op_vals) /
                      (left_max_op_vals - left_min_op_vals)) * 255
    left_op_colors = np.tile(left_opacities[:, np.newaxis], (1, 3))

    t = time()
    left_tex_colors = compute_texture_colors(
        left_parcellation[:, np.newaxis], max_val, min_val=min_val)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    left_colors = left_tex_colors
    left_colors = np.hstack((left_colors, left_opacities[:, np.newaxis]))

    left_hemi_actor = get_hemisphere_actor(fsaverage.pial_left,
                                           colors=left_colors)

    principled_params = manifest_principled(
        left_hemi_actor, subsurface=0, subsurface_color=[0, 0, 0], metallic=0,
        specular=0, specular_tint=0, roughness=0, anisotropic=0,
        anisotropic_direction=[0, 1, .5], sheen=1, sheen_tint=1, clearcoat=0,
        clearcoat_gloss=0)

    opacity = 1.
    left_hemi_actor.GetProperty().SetOpacity(opacity)

    right_max_op_vals = -np.nanmin(right_sulc_points)
    right_min_op_vals = -np.nanmax(right_sulc_points)

    right_opacities = ((-right_sulc_points - right_min_op_vals) /
                       (right_max_op_vals - right_min_op_vals)) * 255

    t = time()
    right_tex_colors = compute_texture_colors(
        right_parcellation[:, np.newaxis], max_val, min_val=min_val)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    right_colors = right_tex_colors
    right_colors = np.hstack((right_colors, right_opacities[:, np.newaxis]))

    right_hemi_actor = get_hemisphere_actor(fsaverage.pial_right,
                                            colors=right_colors)

    _ = manifest_principled(
        right_hemi_actor, subsurface=0, subsurface_color=[0, 0, 0], metallic=0,
        specular=0, specular_tint=0, roughness=0, anisotropic=0,
        anisotropic_direction=[0, 1, .5], sheen=1, sheen_tint=1, clearcoat=0,
        clearcoat_gloss=0)

    right_hemi_actor.GetProperty().SetOpacity(opacity)

    scene.add(left_hemi_actor)
    scene.add(right_hemi_actor)

    view = 'top left'
    if view == 'dorsal' or view == 'top':
        pass
    elif view == 'anterior' or view == 'front':
        scene.roll(180)
        scene.pitch(80)
    elif view == 'left':
        scene.roll(90)
        scene.pitch(80)
    elif view == 'right':
        scene.roll(270)
        scene.pitch(80)
    elif view == 'top left':
        scene.roll(135)
        scene.pitch(80)

    scene.reset_camera()
    scene.reset_clipping_range()
    #scene.zoom(1.6)

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    principled_panel = ui.Panel2D(
        (380, 500), position=(5, 5), color=(.25, .25, .25), opacity=.75,
        align='right')

    panel_label_principled_brdf = ui.TextBlock2D(text='Principled BRDF',
                                                 font_size=18, bold=True)
    slider_label_subsurface = ui.TextBlock2D(text='Subsurface', font_size=16)
    slider_label_metallic = ui.TextBlock2D(text='Metallic', font_size=16)
    slider_label_specular = ui.TextBlock2D(text='Specular', font_size=16)
    slider_label_specular_tint = ui.TextBlock2D(text='Specular Tint',
                                                font_size=16)
    slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)
    slider_label_anisotropic = ui.TextBlock2D(text='Anisotropic', font_size=16)
    slider_label_sheen = ui.TextBlock2D(text='Sheen', font_size=16)
    slider_label_sheen_tint = ui.TextBlock2D(text='Sheen Tint', font_size=16)
    slider_label_clearcoat = ui.TextBlock2D(text='Clearcoat', font_size=16)
    slider_label_clearcoat_gloss = ui.TextBlock2D(text='Clearcoat Gloss',
                                                  font_size=16)

    label_pad_x = .06

    principled_panel.add_element(panel_label_principled_brdf, (.02, .95))
    principled_panel.add_element(slider_label_subsurface, (label_pad_x, .86))
    principled_panel.add_element(slider_label_metallic, (label_pad_x, .77))
    principled_panel.add_element(slider_label_specular, (label_pad_x, .68))
    principled_panel.add_element(slider_label_specular_tint,
                                 (label_pad_x, .59))
    principled_panel.add_element(slider_label_roughness, (label_pad_x, .5))
    principled_panel.add_element(slider_label_anisotropic, (label_pad_x, .41))
    principled_panel.add_element(slider_label_sheen, (label_pad_x, .32))
    principled_panel.add_element(slider_label_sheen_tint, (label_pad_x, .23))
    principled_panel.add_element(slider_label_clearcoat, (label_pad_x, .14))
    principled_panel.add_element(slider_label_clearcoat_gloss,
                                 (label_pad_x, .05))

    length = 200
    text_template = '{value:.1f}'

    slider_slice_subsurface = ui.LineSlider2D(
        initial_value=principled_params['subsurface'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_metallic = ui.LineSlider2D(
        initial_value=principled_params['metallic'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_specular = ui.LineSlider2D(
        initial_value=principled_params['specular'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_specular_tint = ui.LineSlider2D(
        initial_value=principled_params['specular_tint'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=principled_params['roughness'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_anisotropic = ui.LineSlider2D(
        initial_value=principled_params['anisotropic'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_sheen = ui.LineSlider2D(
        initial_value=principled_params['sheen'], max_value=1, length=length,
        text_template=text_template)
    slider_slice_sheen_tint = ui.LineSlider2D(
        initial_value=principled_params['sheen_tint'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_clearcoat = ui.LineSlider2D(
        initial_value=principled_params['clearcoat'], max_value=1,
        length=length, text_template=text_template)
    slider_slice_clearcoat_gloss = ui.LineSlider2D(
        initial_value=principled_params['clearcoat_gloss'], max_value=1,
        length=length, text_template=text_template)

    slider_slice_subsurface.on_change = change_slice_subsurface
    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_specular.on_change = change_slice_specular
    slider_slice_specular_tint.on_change = change_slice_specular_tint
    slider_slice_roughness.on_change = change_slice_roughness
    slider_slice_anisotropic.on_change = change_slice_anisotropic
    slider_slice_sheen.on_change = change_slice_sheen
    slider_slice_sheen_tint.on_change = change_slice_sheen_tint
    slider_slice_clearcoat.on_change = change_slice_clearcoat
    slider_slice_clearcoat_gloss.on_change = change_slice_clearcoat_gloss

    slice_pad_x = .4

    principled_panel.add_element(slider_slice_subsurface, (slice_pad_x, .86))
    principled_panel.add_element(slider_slice_metallic, (slice_pad_x, .77))
    principled_panel.add_element(slider_slice_specular, (slice_pad_x, .68))
    principled_panel.add_element(slider_slice_specular_tint,
                                 (slice_pad_x, .59))
    principled_panel.add_element(slider_slice_roughness, (slice_pad_x, .5))
    principled_panel.add_element(slider_slice_anisotropic, (slice_pad_x, .41))
    principled_panel.add_element(slider_slice_sheen, (slice_pad_x, .32))
    principled_panel.add_element(slider_slice_sheen_tint, (slice_pad_x, .23))
    principled_panel.add_element(slider_slice_clearcoat, (slice_pad_x, .14))
    principled_panel.add_element(slider_slice_clearcoat_gloss,
                                 (slice_pad_x, .05))

    scene.add(principled_panel)

    params_panel = ui.Panel2D((380, 400), position=(5, 510),
                              color=(.25, .25, .25), opacity=.75,
                              align='right')

    panel_label_params = ui.TextBlock2D(text='Parameters', font_size=18,
                                        bold=True)
    section_label_subsurf_color = ui.TextBlock2D(text='Subsurface Color',
                                                 font_size=16, bold=True)
    slider_label_subsurf_r = ui.TextBlock2D(text='R', font_size=16)
    slider_label_subsurf_g = ui.TextBlock2D(text='G', font_size=16)
    slider_label_subsurf_b = ui.TextBlock2D(text='B', font_size=16)
    section_label_aniso_dir = ui.TextBlock2D(text='Anisotropic Direction',
                                             font_size=16, bold=True)
    slider_label_aniso_x = ui.TextBlock2D(text='X', font_size=16)
    slider_label_aniso_y = ui.TextBlock2D(text='Y', font_size=16)
    slider_label_aniso_z = ui.TextBlock2D(text='Z', font_size=16)

    params_panel.add_element(panel_label_params, (.02, .94))
    params_panel.add_element(section_label_subsurf_color, (.04, .83))
    params_panel.add_element(slider_label_subsurf_r, (label_pad_x, .72))
    params_panel.add_element(slider_label_subsurf_g, (label_pad_x, .61))
    params_panel.add_element(slider_label_subsurf_b, (label_pad_x, .5))
    params_panel.add_element(section_label_aniso_dir, (.04, .39))
    params_panel.add_element(slider_label_aniso_x, (label_pad_x, .28))
    params_panel.add_element(slider_label_aniso_y, (label_pad_x, .17))
    params_panel.add_element(slider_label_aniso_z, (label_pad_x, .06))

    slider_slice_subsurf_r = ui.LineSlider2D(
        initial_value=principled_params['subsurface_color'][0], max_value=1,
        length=length, text_template=text_template)
    slider_slice_subsurf_g = ui.LineSlider2D(
        initial_value=principled_params['subsurface_color'][1], max_value=1,
        length=length, text_template=text_template)
    slider_slice_subsurf_b = ui.LineSlider2D(
        initial_value=principled_params['subsurface_color'][2], max_value=1,
        length=length, text_template=text_template)
    slider_slice_aniso_x = ui.LineSlider2D(
        initial_value=principled_params['anisotropic_direction'][0],
        min_value=-1, max_value=1, length=length, text_template=text_template)
    slider_slice_aniso_y = ui.LineSlider2D(
        initial_value=principled_params['anisotropic_direction'][1],
        min_value=-1, max_value=1, length=length, text_template=text_template)
    slider_slice_aniso_z = ui.LineSlider2D(
        initial_value=principled_params['anisotropic_direction'][2],
        min_value=-1, max_value=1, length=length, text_template=text_template)

    slider_slice_subsurf_r.on_change = change_slice_subsurf_r
    slider_slice_subsurf_g.on_change = change_slice_subsurf_g
    slider_slice_subsurf_b.on_change = change_slice_subsurf_b
    slider_slice_aniso_x.on_change = change_slice_aniso_x
    slider_slice_aniso_y.on_change = change_slice_aniso_y
    slider_slice_aniso_z.on_change = change_slice_aniso_z

    params_panel.add_element(slider_slice_subsurf_r, (slice_pad_x, .72))
    params_panel.add_element(slider_slice_subsurf_g, (slice_pad_x, .61))
    params_panel.add_element(slider_slice_subsurf_b, (slice_pad_x, .5))
    params_panel.add_element(slider_slice_aniso_x, (slice_pad_x, .28))
    params_panel.add_element(slider_slice_aniso_y, (slice_pad_x, .17))
    params_panel.add_element(slider_slice_aniso_z, (slice_pad_x, .06))

    scene.add(params_panel)

    control_panel = ui.Panel2D((380, 80), position=(5, 915),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = ui.TextBlock2D(text='Control', font_size=18,
                                         bold=True)
    slider_label_opacity = ui.TextBlock2D(text='Opacity', font_size=16)

    control_panel.add_element(panel_label_control, (.02, .7))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .3))

    slider_slice_opacity = ui.LineSlider2D(
        initial_value=opacity, max_value=1, length=length,
        text_template=text_template)

    slider_slice_opacity.on_change = change_slice_opacity

    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .3))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()

    #window.record(scene, out_path='sheen_atlas_2.png', size=(1920, 1080),
    #              magnification=4)
