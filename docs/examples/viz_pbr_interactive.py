"""
===============================================
Interactive PBR demo
===============================================

This is a demonstration of how Physically-Based Rendering (PBR) can be used to
simulate different materials.

Let's start by importing the necessary modules:
"""

from fury import actor, material, ui, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.utils import (normals_from_actor, tangents_to_actor,
                        tangents_from_direction_of_anisotropy)


###############################################################################
# The following functions will help us to manage the sliders events.


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


###############################################################################
# Last, but not least, we define the following function to help us to
# reposition the UI elements every time we resize the window.


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        control_panel.re_align(size_change)


###############################################################################
# Let's fetch a skybox texture from the FURY data repository.

fetch_viz_cubemaps()

###############################################################################
# The following function returns the full path of the 6 images composing the
# skybox.

textures = read_viz_cubemap('skybox')

###############################################################################
# Now that we have the location of the textures, let's load them and create a
# Cube Map Texture object.

cubemap = load_cubemap_texture(textures)

###############################################################################
# The Scene object in FURY can handle cube map textures and extract light
# information from them, so it can be used to create more plausible materials
# interactions. The ``skybox`` parameter takes as input a cube map texture and
# performs the previously described process.

scene = window.Scene(skybox=cubemap)

###############################################################################
# With the scene created, we can then populate it. In this demo we will only
# add a sphere actor.

sphere = actor.sphere([[0, 0, 0]], (.7, .7, .7), radii=2, theta=64, phi=64)

###############################################################################
# The direction of anisotropy (DoA) defines the direction at which all the
# tangents of our actor are pointing.

doa = [0, 1, .5]

###############################################################################
# The following process gets the normals of the actor and computes the tangents
# that are aligned to the provided DoA. Then it registers those tangents to the
# actor.

normals = normals_from_actor(sphere)
tangents = tangents_from_direction_of_anisotropy(normals, doa)
tangents_to_actor(sphere, tangents)

###############################################################################
# With the tangents computed and in place, we have all the elements needed to
# add some material properties to the actor.

pbr_params = material.manifest_pbr(sphere)

###############################################################################
# Our actor is now ready to be added to the scene.

scene.add(sphere)

###############################################################################
# Let's setup now the window and the UI.

show_m = window.ShowManager(scene=scene, size=(1920, 1080), reset_camera=False,
                            order_transparent=True)


###############################################################################
# We will create one single panel with all of our labels and sliders.

control_panel = ui.Panel2D(
    (400, 500), position=(5, 5), color=(.25, .25, .25), opacity=.75,
    align='right')

###############################################################################
# By using our previously defined function, we can easily create all the labels
# we need for this demo. And then add them to the panel.

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
slider_label_coat_strength = ui.TextBlock2D(text='Coat Strength', font_size=16)
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

###############################################################################
# Our sliders are created and added to the panel in the following way.

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

###############################################################################
# Notice that we are defining a range of [-1, 1] for the DoA. This is because
# within that range we cover all the possible 3D directions needed to align the
# tangents.

slider_slice_anisotropy_direction_x = ui.LineSlider2D(
    initial_value=doa[0], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_direction_y = ui.LineSlider2D(
    initial_value=doa[1], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_direction_z = ui.LineSlider2D(
    initial_value=doa[2], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')

###############################################################################
# Another special case are the Index of Refraction (IoR) sliders. In these
# cases, the values are defined in the range [1, 2.3] according to the
# documentation of the material.

slider_slice_base_ior = ui.LineSlider2D(
    initial_value=pbr_params.base_ior, min_value=1, max_value=2.3,
    length=195, text_template='{value:.02f}')
slider_slice_coat_ior = ui.LineSlider2D(
    initial_value=pbr_params.coat_ior, min_value=1, max_value=2.3,
    length=195, text_template='{value:.02f}')

###############################################################################
# Let's add the event handlers functions to the corresponding sliders.

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

###############################################################################
# And then add the sliders to the panel.

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

###############################################################################
# Consequently, we add the panel to the scene.

scene.add(control_panel)

###############################################################################
# Previously we defined a function to help us when we resize the window, so
# let's capture the current size and add our helper function as a
# `window_callback` to the window.

size = scene.GetSize()

show_m.add_window_callback(win_callback)

###############################################################################
# Finally, let's visualize our demo.

interactive = False
if interactive:
    show_m.start()

window.record(scene, size=(1920, 1080), out_path="viz_pbr_interactive.png")
