from dipy.data import get_fnames
from fury import actor, window
from fury.data import read_viz_textures
from fury.io import load_polydata
from fury.utils import get_actor_from_polydata


import numpy as np
import os
import vtk


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


if __name__ == '__main__':
    brain_lh = get_fnames(name='fury_surface')
    polydata = load_polydata(brain_lh)

    surface_actor = get_actor_from_polydata(polydata)

    surface_actor.GetProperty().SetInterpolationToPBR()

    # Lets use a rough metallic surface
    metallic_coef = 1.
    roughness_coef = 0.

    surface_actor.GetProperty().SetMetallic(metallic_coef)
    surface_actor.GetProperty().SetRoughness(roughness_coef)

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

    scene.add(surface_actor)
    scene.add(skybox_actor)

    window.show(scene)
