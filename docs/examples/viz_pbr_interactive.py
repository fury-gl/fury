from fury import actor, material, window
from fury.io import load_cubemap_texture
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
import numpy as np

fetch_viz_cubemaps()

textures = read_viz_cubemap('skybox')

cubemap = load_cubemap_texture(textures)

scene = window.Scene(skybox_tex=cubemap, render_skybox=True)

#skybox.RepeatOff()
#skybox.EdgeClampOn()

sphere = actor.sphere([[0, 0, 0]], (.7, .7, .7), radii=2, theta=64, phi=64)
material.manifest_pbr(sphere)

scene.add(sphere)

window.show(scene)
