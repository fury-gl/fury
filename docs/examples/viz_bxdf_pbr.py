from dipy.data import get_fnames
from fury import actor, ui, window
from fury.data import fetch_viz_models, read_viz_models, read_viz_textures
from fury.io import load_polydata
from fury.utils import (get_actor_from_polydata, get_polydata_colors,
                        get_polydata_triangles, get_polydata_vertices,
                        normals_from_v_f, set_polydata_colors,
                        set_polydata_normals)
from fury.shaders import add_shader_callback, load, shader_to_actor
from scipy.spatial import Delaunay


import math
import numpy as np
import os
import random
import vtk


def build_label(text, font_size=16, color=(1, 1, 1), bold=False, italic=False,
                shadow=False):
    label = ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = bold
    label.italic = italic
    label.shadow = shadow
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = color
    return label


def change_slice_subsurface(slider):
    global subsurface
    subsurface = slider._value


def change_slice_metallic(slider):
    global obj_actor
    obj_actor.GetProperty().SetMetallic(slider._value)


def change_slice_specular(slider):
    global obj_actor
    obj_actor.GetProperty().SetSpecular(slider._value)


def change_slice_specular_tint(slider):
    global specular_tint
    specular_tint = slider._value


def change_slice_roughness(slider):
    global obj_actor
    obj_actor.GetProperty().SetRoughness(slider._value)


def change_slice_anisotropic(slider):
    global anisotropic
    anisotropic = slider._value


def change_slice_sheen(slider):
    global sheen
    sheen = slider._value


def change_slice_sheen_tint(slider):
    global sheen_tint
    sheen_tint = slider._value


def change_slice_clearcoat(slider):
    global clearcoat
    clearcoat = slider._value


def change_slice_clearcoat_gloss(slider):
    global clearcoat_gloss
    clearcoat_gloss = slider._value


def change_slice_opacity(slider):
    global obj_actor
    obj_actor.GetProperty().SetOpacity(slider._value)


def get_cubemap(files_names):
    texture = vtk.vtkTexture()
    texture.CubeMapOn()
    for idx, fn in enumerate(files_names):
        if not os.path.isfile(fn):
            print('Nonexistent texture file:', fn)
            return texture
        else:
            # Read the images
            reader_factory = vtk.vtkImageReader2Factory()
            img_reader = reader_factory.CreateImageReader2(fn)
            img_reader.SetFileName(fn)

            flip = vtk.vtkImageFlip()
            flip.SetInputConnection(img_reader.GetOutputPort())
            flip.SetFilteredAxis(1)  # flip y axis
            texture.SetInputConnection(idx, flip.GetOutputPort(0))
    return texture


def obj_brain():
    brain_lh = get_fnames(name='fury_surface')
    polydata = load_polydata(brain_lh)
    verts = get_polydata_vertices(polydata)
    faces = get_polydata_triangles(polydata)
    normals = normals_from_v_f(verts, faces)
    set_polydata_normals(polydata, normals)
    return get_actor_from_polydata(polydata)


def obj_model(model='glyptotek.vtk', color=None):
    if model != 'glyptotek.vtk':
        fetch_viz_models()
    model = read_viz_models(model)
    polydata = load_polydata(model)
    if color is not None:
        color = np.asarray([color]) * 255
        colors = get_polydata_colors(polydata)
        if colors is not None:
            num_vertices = colors.shape[0]
            new_colors = np.repeat(color, num_vertices, axis=0)
            colors[:, :] = new_colors
        else:
            vertices = get_polydata_vertices(polydata)
            num_vertices = vertices.shape[0]
            new_colors = np.repeat(color, num_vertices, axis=0)
            set_polydata_colors(polydata, new_colors)
    return get_actor_from_polydata(polydata)


def obj_spheres(radii=2, theta=32, phi=32):
    centers = [[-5, 5, 0], [0, 5, 0], [5, 5, 0], [-5, 0, 0], [0, 0, 0],
               [5, 0, 0], [-5, -5, 0], [0, -5, 0], [5, -5, 0]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0],
              [0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    return actor.sphere(centers, colors, radii=radii, theta=theta, phi=phi)


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
    normals = normals_from_v_f(vertices, faces)
    #polydata = vtk.vtkPolyData()
    #polydata.DeepCopy(surface_actor.GetMapper().GetInput())
    polydata = surface_actor.GetMapper().GetInput()
    set_polydata_normals(polydata, normals)
    #surface_actor = get_actor_from_polydata(polydata)
    return surface_actor


def uniforms_callback(_caller, _event, calldata=None):
    global anisotropic, clearcoat, clearcoat_gloss, sheen, sheen_tint, \
        specular_tint, subsurface
    if calldata is not None:
        calldata.SetUniformf('subsurface', subsurface)
        calldata.SetUniformf('specularTint', specular_tint)
        calldata.SetUniformf('anisotropic', anisotropic)
        calldata.SetUniformf('sheen', sheen)
        calldata.SetUniformf('sheenTint', sheen_tint)
        calldata.SetUniformf('clearcoat', clearcoat)
        calldata.SetUniformf('clearcoatGloss', clearcoat_gloss)


def win_callback(obj, event):
    global control_panel, pbr_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        pbr_panel.re_align(size_change)
        control_panel.re_align(size_change)


if __name__ == '__main__':
    global anisotropic, clearcoat, clearcoat_gloss, control_panel, obj_actor, \
        pbr_panel, sheen, sheen_tint, size, specular_tint, subsurface

    #obj_actor = obj_brain()
    #obj_actor = obj_surface()
    #obj_actor = obj_model(model='suzanne.obj', color=(0, 1, 1))
    obj_actor = obj_model(model='glyptotek.vtk', color=(0, 1, 1))
    #obj_actor = obj_spheres()

    subsurface = .0
    metallic = .0
    specular = .0
    specular_tint = .0
    roughness = .0
    anisotropic = .0
    sheen = .0
    sheen_tint = .0
    clearcoat = .0
    clearcoat_gloss = .0

    obj_actor.GetProperty().SetInterpolationToPBR()
    obj_actor.GetProperty().SetMetallic(metallic)
    obj_actor.GetProperty().SetRoughness(roughness)

    # NOTE: Specular parameters don't seem to work
    #specular_color = vtk.vtkNamedColors().GetColor3d('Blue')
    obj_actor.GetProperty().SetSpecular(specular)
    #obj_actor.GetProperty().SetSpecularPower(specular_tint)
    #obj_actor.GetProperty().SetSpecularColor(specular_color)

    opacity = 1.
    obj_actor.GetProperty().SetOpacity(opacity)

    add_shader_callback(obj_actor, uniforms_callback)

    fs_dec_code = load('bxdf_dec.frag')
    fs_impl_code = load('bxdf_impl.frag')

    #shader_to_actor(obj_actor, 'vertex', debug=True)
    shader_to_actor(obj_actor, 'fragment', decl_code=fs_dec_code)
    shader_to_actor(obj_actor, 'fragment', impl_code=fs_impl_code,
                    block='light', debug=False)

    cubemap_fns = [read_viz_textures('skybox-px.jpg'),
                   read_viz_textures('skybox-nx.jpg'),
                   read_viz_textures('skybox-py.jpg'),
                   read_viz_textures('skybox-ny.jpg'),
                   read_viz_textures('skybox-pz.jpg'),
                   read_viz_textures('skybox-nz.jpg')]

    # Load the cube map
    cubemap = get_cubemap(cubemap_fns)

    # Load the skybox
    skybox = get_cubemap(cubemap_fns)
    skybox.InterpolateOn()
    skybox.RepeatOff()
    skybox.EdgeClampOn()

    skybox_actor = vtk.vtkSkybox()
    skybox_actor.SetTexture(skybox)

    scene = window.Scene()

    scene.UseImageBasedLightingOn()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        scene.SetEnvironmentTexture(cubemap)
    else:
        scene.SetEnvironmentCubeMap(cubemap)

    scene.add(obj_actor)
    scene.add(skybox_actor)

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, reset_camera=False,
                                order_transparent=True)
    show_m.initialize()

    pbr_panel = ui.Panel2D((320, 500), position=(-25, 5),
                           color=(.25, .25, .25), opacity=.75, align='right')

    panel_label_principled_brdf = build_label('"Principled" BRDF',
                                              font_size=18, bold=True)
    slider_label_subsurface = build_label('Subsurface')
    slider_label_metallic = build_label('Metallic')
    slider_label_specular = build_label('Specular')
    slider_label_specular_tint = build_label('Specular Tint')
    slider_label_roughness = build_label('Roughness')
    slider_label_anisotropic = build_label('Anisotropic')
    slider_label_sheen = build_label('Sheen')
    slider_label_sheen_tint = build_label('Sheen Tint')
    slider_label_clearcoat = build_label('Clearcoat')
    slider_label_clearcoat_gloss = build_label('Clearcoat Gloss')

    label_pad_x = .06

    pbr_panel.add_element(panel_label_principled_brdf, (.02, .95))
    pbr_panel.add_element(slider_label_subsurface, (label_pad_x, .86))
    pbr_panel.add_element(slider_label_metallic, (label_pad_x, .77))
    pbr_panel.add_element(slider_label_specular, (label_pad_x, .68))
    pbr_panel.add_element(slider_label_specular_tint, (label_pad_x, .59))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .5))
    pbr_panel.add_element(slider_label_anisotropic, (label_pad_x, .41))
    pbr_panel.add_element(slider_label_sheen, (label_pad_x, .32))
    pbr_panel.add_element(slider_label_sheen_tint, (label_pad_x, .23))
    pbr_panel.add_element(slider_label_clearcoat, (label_pad_x, .14))
    pbr_panel.add_element(slider_label_clearcoat_gloss, (label_pad_x, .05))

    length = 150
    text_template = '{value:.1f}'

    slider_slice_subsurface = ui.LineSlider2D(
        initial_value=subsurface, max_value=1, length=length,
        text_template=text_template)
    slider_slice_metallic = ui.LineSlider2D(
        initial_value=metallic, max_value=1, length=length,
        text_template=text_template)
    slider_slice_specular = ui.LineSlider2D(
        initial_value=specular, max_value=1, length=length,
        text_template=text_template)
    slider_slice_specular_tint = ui.LineSlider2D(
        initial_value=specular_tint, max_value=1, length=length,
        text_template=text_template)
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=roughness, max_value=1, length=length,
        text_template=text_template)
    slider_slice_anisotropic = ui.LineSlider2D(
        initial_value=anisotropic, max_value=1, length=length,
        text_template=text_template)
    slider_slice_sheen = ui.LineSlider2D(
        initial_value=sheen, max_value=1, length=length,
        text_template=text_template)
    slider_slice_sheen_tint = ui.LineSlider2D(
        initial_value=sheen_tint, max_value=1, length=length,
        text_template=text_template)
    slider_slice_clearcoat = ui.LineSlider2D(
        initial_value=clearcoat, max_value=1, length=length,
        text_template=text_template)
    slider_slice_clearcoat_gloss = ui.LineSlider2D(
        initial_value=clearcoat_gloss, max_value=1, length=length,
        text_template=text_template)

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

    slice_pad_x = .46

    pbr_panel.add_element(slider_slice_subsurface, (slice_pad_x, .86))
    pbr_panel.add_element(slider_slice_metallic, (slice_pad_x, .77))
    pbr_panel.add_element(slider_slice_specular, (slice_pad_x, .68))
    pbr_panel.add_element(slider_slice_specular_tint, (slice_pad_x, .59))
    pbr_panel.add_element(slider_slice_roughness, (slice_pad_x, .5))
    pbr_panel.add_element(slider_slice_anisotropic, (slice_pad_x, .41))
    pbr_panel.add_element(slider_slice_sheen, (slice_pad_x, .32))
    pbr_panel.add_element(slider_slice_sheen_tint, (slice_pad_x, .23))
    pbr_panel.add_element(slider_slice_clearcoat, (slice_pad_x, .14))
    pbr_panel.add_element(slider_slice_clearcoat_gloss, (slice_pad_x, .05))

    scene.add(pbr_panel)

    control_panel = ui.Panel2D((320, 80), position=(-25, 510),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = build_label('Control', font_size=18,
                                               bold=True)
    slider_label_opacity = build_label('Opacity')

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
