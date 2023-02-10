#from dipy.data import get_fnames
from fury import actor, ui, window
from fury.data import (fetch_viz_cubemaps, fetch_viz_models, read_viz_cubemap,
                       read_viz_models)
from fury.io import load_cubemap_texture, load_polydata
from fury.lib import ImageData, Texture, numpy_support
from fury.material import manifest_principled
from fury.utils import (get_actor_from_polydata, get_polydata_colors,
                        get_polydata_normals, get_polydata_triangles,
                        get_polydata_vertices, normals_from_v_f, rotate,
                        set_polydata_colors, set_polydata_normals)
from scipy.spatial import Delaunay


import math
import numpy as np
import random


def change_slice_subsurface(slider):
    global principled_params
    principled_params['subsurface'] = slider.value


def change_slice_metallic(slider):
    global principled_params
    principled_params['metallic'] = slider.value


def change_slice_specular(slider):
    global obj_actor, principled_params
    principled_params['specular'] = slider.value
    obj_actor.GetProperty().SetSpecular(slider.value)


def change_slice_specular_tint(slider):
    global principled_params
    principled_params['specular_tint'] = slider.value


def change_slice_roughness(slider):
    global principled_params
    principled_params['roughness'] = slider.value


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
    global obj_actor
    obj_actor.GetProperty().SetOpacity(slider.value)


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


"""
def obj_brain():
    brain_lh = get_fnames(name='fury_surface')
    polydata = load_polydata(brain_lh)
    verts = get_polydata_vertices(polydata)
    faces = get_polydata_triangles(polydata)
    normals = normals_from_v_f(verts, faces)
    set_polydata_normals(polydata, normals)
    return get_actor_from_polydata(polydata)
"""


def obj_model(model='glyptotek.vtk', colors=None):
    if model != 'glyptotek.vtk':
        fetch_viz_models()
    model = read_viz_models(model)
    polydata = load_polydata(model)
    if colors is not None:
        poly_colors = get_polydata_colors(polydata)
        if type(colors) is tuple:
            color = np.asarray([colors]) * 255
            if poly_colors is not None:
                num_verts = poly_colors.shape[0]
            else:
                verts = get_polydata_vertices(polydata)
                num_verts = verts.shape[0]
            new_colors = np.repeat(color, num_verts, axis=0)
        elif colors.lower() == 'normals':
            normals = get_polydata_normals(polydata)
            if normals is not None:
                new_colors = (normals + 1) / 2 * 255
        else:
            # TODO: Check ndarray
            new_colors = colors * 255
        # Either replace or set
        if poly_colors is not None:
            # Replace original colors
            poly_colors[:, :] = new_colors
        else:
            # Set new color array
            set_polydata_colors(polydata, new_colors)
    return get_actor_from_polydata(polydata)


def obj_spheres(radii=2, theta=32, phi=32):
    centers = np.array([[-5, 5, 0], [0, 5, 0], [5, 5, 0], [-5, 0, 0],
                        [0, 0, 0], [5, 0, 0], [-5, -5, 0], [0, -5, 0],
                        [5, -5, 0]])
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0],
              [0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    return actor.sphere(centers, colors, radii=radii, theta=theta, phi=phi,
                        use_primitive=False)


def obj_surface():
    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = - math.sin(i) * math.cos(j)
            fact2 = - math.exp(abs(1 - math.sqrt(i ** 2 + j ** 2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])
    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices
    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, "butterfly", "loop"]
    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                surface_actor = actor.surface(vertices, faces=face,
                                              colors=color, smooth=smooth_type)
    # TODO: Report bug with polydata content not updated
    polydata = surface_actor.GetMapper().GetInput()
    vertices = get_polydata_vertices(polydata)
    faces = get_polydata_triangles(polydata)
    normals = normals_from_v_f(vertices, faces)
    set_polydata_normals(polydata, normals)
    return surface_actor


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
    global control_panel, obj_actor, params_panel, principled_panel, \
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

    #obj_actor = obj_brain()
    #obj_actor = obj_surface()
    #obj_actor = obj_model(model='suzanne.obj', colors=(0, 1, 1))
    #obj_actor = obj_model(model='glyptotek.vtk', colors=(.75, .48, .34))
    #obj_actor = obj_model(model='glyptotek.vtk', colors=(1, 1, 0))
    #obj_actor = obj_model(model='glyptotek.vtk', colors='normals')
    #obj_actor = obj_model(model='glyptotek.vtk')
    obj_actor = obj_spheres()

    # Glyptotek's rotation
    #rotate(obj_actor, rotation=(-145, 0, 0, 1))
    #rotate(obj_actor, rotation=(-70, 1, 0, 0))

    # Actor rotation for brudslojan texture
    #rotate(obj_actor, rotation=(-110, 0, 1, 0))

    scene.add(obj_actor)

    scene.reset_camera()
    #scene.zoom(1.9)  # Glyptotek's zoom

    principled_params = manifest_principled(
        obj_actor, subsurface=0, subsurface_color=[0, 0, 0], metallic=0,
        specular=0, specular_tint=0, roughness=0, anisotropic=0,
        anisotropic_direction=[0, 1, .5], sheen=0, sheen_tint=0, clearcoat=0,
        clearcoat_gloss=0)

    opacity = 1.
    obj_actor.GetProperty().SetOpacity(opacity)

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
