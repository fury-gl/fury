from fury import actor, material, ui, window
from fury.io import load_cubemap_texture
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.utils import (normals_from_actor, tangents_to_actor,
                        tangents_from_direction_of_anisotropy)


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


def change_slice_metallic(slider):
    global pbr_params, sphere
    pbr_params['metallic'] = slider.value
    sphere.GetProperty().SetMetallic(pbr_params['metallic'])


def change_slice_roughness(slider):
    global pbr_params, sphere
    pbr_params['roughness'] = slider.value
    sphere.GetProperty().SetRoughness(pbr_params['roughness'])


def change_slice_anisotropy(slider):
    global pbr_params, sphere
    pbr_params['anisotropy'] = slider.value
    sphere.GetProperty().SetAnisotropy(pbr_params['anisotropy'])


def change_slice_anisotropy_direction_x(slider):
    global doa, normals, sphere
    doa[0] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_direction_y(slider):
    global doa, normals, sphere
    doa[1] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_direction_z(slider):
    global doa, normals, sphere
    doa[2] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_rotation(slider):
    global pbr_params, sphere
    pbr_params['anisotropy_rotation'] = slider.value
    sphere.GetProperty().SetAnisotropyRotation(
        pbr_params['anisotropy_rotation'])


def change_slice_coat_strength(slider):
    global pbr_params, sphere
    pbr_params['coat_strength'] = slider.value
    sphere.GetProperty().SetCoatStrength(pbr_params['coat_strength'])


def change_slice_coat_roughness(slider):
    global pbr_params, sphere
    pbr_params['coat_roughness'] = slider.value
    sphere.GetProperty().SetCoatRoughness(pbr_params['coat_roughness'])


def change_slice_base_ior(slider):
    global pbr_params, sphere
    pbr_params['base_ior'] = slider.value
    sphere.GetProperty().SetBaseIOR(pbr_params['base_ior'])


def change_slice_coat_ior(slider):
    global pbr_params, sphere
    pbr_params['coat_ior'] = slider.value
    sphere.GetProperty().SetCoatIOR(pbr_params['coat_ior'])


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        control_panel.re_align(size_change)


fetch_viz_cubemaps()

textures = read_viz_cubemap('skybox')

cubemap = load_cubemap_texture(textures)

scene = window.Scene(skybox_tex=cubemap, render_skybox=True)

sphere = actor.sphere([[0, 0, 0]], (.7, .7, .7), radii=2, theta=64, phi=64)

doa = [0, 1, .5]

normals = normals_from_actor(sphere)
tangents = tangents_from_direction_of_anisotropy(normals, doa)
tangents_to_actor(sphere, tangents)

pbr_params = material.manifest_pbr(sphere)

scene.add(sphere)

show_m = window.ShowManager(scene=scene, size=(1920, 1080), reset_camera=False,
                            order_transparent=True)
show_m.initialize()

control_panel = ui.Panel2D(
    (400, 500), position=(5, 5), color=(.25, .25, .25), opacity=.75,
    align='right')

slider_label_metallic = build_label('Metallic')
slider_label_roughness = build_label('Roughness')
slider_label_anisotropy = build_label('Anisotropy')
slider_label_anisotropy_rotation = build_label('Anisotropy Rotation')
slider_label_anisotropy_direction_x = build_label('Anisotropy Direction X')
slider_label_anisotropy_direction_y = build_label('Anisotropy Direction Y')
slider_label_anisotropy_direction_z = build_label('Anisotropy Direction Z')
slider_label_coat_strength = build_label('Coat Strength')
slider_label_coat_roughness = build_label('Coat Roughness')
slider_label_base_ior = build_label('Base IoR')
slider_label_coat_ior = build_label('Coat IoR')

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
    initial_value=pbr_params['metallic'], max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_roughness = ui.LineSlider2D(
    initial_value=pbr_params['roughness'], max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy = ui.LineSlider2D(
    initial_value=pbr_params['anisotropy'], max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_rotation = ui.LineSlider2D(
    initial_value=pbr_params['anisotropy_rotation'], max_value=1, length=195,
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

slider_slice_coat_strength = ui.LineSlider2D(
    initial_value=pbr_params['coat_strength'], max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_coat_roughness = ui.LineSlider2D(
    initial_value=pbr_params['coat_roughness'], max_value=1, length=195,
    text_template='{value:.1f}')

slider_slice_base_ior = ui.LineSlider2D(
    initial_value=pbr_params['base_ior'], min_value=1, max_value=2.3,
    length=195, text_template='{value:.02f}')
slider_slice_coat_ior = ui.LineSlider2D(
    initial_value=pbr_params['coat_ior'], min_value=1, max_value=2.3,
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
