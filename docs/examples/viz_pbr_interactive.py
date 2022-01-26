from fury import actor, material, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.lib import ImageFlip, ImageReader2Factory, Texture
import numpy as np
import os


def get_cubemap_texture(file_names, interpolate_on=True, mipmap_on=True):
    texture = Texture()
    texture.CubeMapOn()
    for idx, fn in enumerate(file_names):
        if not os.path.isfile(fn):
            raise FileNotFoundError(fn)
        else:
            # Read the images
            reader_factory = ImageReader2Factory()
            img_reader = reader_factory.CreateImageReader2(fn)
            img_reader.SetFileName(fn)

            flip = ImageFlip()
            flip.SetInputConnection(img_reader.GetOutputPort())
            flip.SetFilteredAxis(1)  # flip y axis
            texture.SetInputConnection(idx, flip.GetOutputPort(0))
    if interpolate_on:
        texture.InterpolateOn()
    if mipmap_on:
        texture.MipmapOn()
    return texture


fetch_viz_cubemaps()

textures = read_viz_cubemap('skybox')

cubemap = get_cubemap_texture(textures)

scene = window.Scene(skybox_tex=cubemap, render_skybox=True)

#skybox.RepeatOff()
#skybox.EdgeClampOn()

sphere = actor.sphere([[0, 0, 0]], (.7, .7, .7), radii=2, theta=64, phi=64)
material.manifest_pbr(sphere)

scene.add(sphere)

window.show(scene)
