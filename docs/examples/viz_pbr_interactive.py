from fury import actor, material, window
from fury.data import fetch_viz_textures, read_viz_textures
from fury.lib import (VTK_9_PLUS, VTK_UNSIGNED_INT, ImageFlip,
                      ImageReader2Factory, Skybox, Texture, numpy_support)
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


# TODO: Fetch skybox
# TODO: Create wrapper function with name order
texture_name = 'skybox'
cubemap_fns = [read_viz_textures(texture_name + '-px.jpg'),
               read_viz_textures(texture_name + '-nx.jpg'),
               read_viz_textures(texture_name + '-py.jpg'),
               read_viz_textures(texture_name + '-ny.jpg'),
               read_viz_textures(texture_name + '-pz.jpg'),
               read_viz_textures(texture_name + '-nz.jpg')]

cubemap = get_cubemap_texture(cubemap_fns)

scene = window.Scene()

print(scene.GetUseImageBasedLighting())

# TODO: Add to Scene constructor
scene.UseImageBasedLightingOn()
if VTK_9_PLUS:
    scene.SetEnvironmentTexture(cubemap)
else:
    scene.SetEnvironmentCubeMap(cubemap)

print(scene.GetUseImageBasedLighting())

#skybox.RepeatOff()
#skybox.EdgeClampOn()

skybox = Skybox()
skybox.SetTexture(cubemap)

scene.add(skybox)

sphere = actor.sphere([[0, 0, 0]], (.7, .7, .7), radii=2, theta=64, phi=64)
polydata = sphere.GetMapper().GetInput()

print(polydata.GetFieldData())

# TODO: get_polydata_field/field_from_actor
print(polydata.GetFieldData().GetArray('Uses IBL'))

# TODO: add_polydata_field/field_to_actor
field = np.array([[True]])
field_name = 'Uses IBL'
array_type = VTK_UNSIGNED_INT
vtk_field = numpy_support.numpy_to_vtk(field, deep=True, array_type=array_type)
vtk_field.SetName(field_name)
polydata.GetFieldData().AddArray(vtk_field)

print(polydata.GetFieldData())

# TODO: get_polydata_field/field_from_actor
print(numpy_support.vtk_to_numpy(polydata.GetFieldData().GetArray('Uses IBL')))

scene.add(sphere)

window.show(scene)
