import numpy as np
from fury import window, actor, ui, io, utils
from fury.shaders import shader_to_actor
from fury.data.fetcher import fetch_viz_models, read_viz_models,\
                              fetch_viz_textures, read_viz_textures
import vtk

scene = window.Scene()

fetch_viz_models()
dragon = read_viz_models('dragon.obj')
dragon = io.load_polydata(dragon)
dragon = utils.get_polymapper_from_polydata(dragon)
dragon = utils.get_actor_from_polymapper(dragon)

dragon.GetProperty().SetSpecular(0.8)
dragon.GetProperty().SetSpecularPower(20)
dragon.GetProperty().SetDiffuse(0.1)

scene.add(dragon)

fetch_viz_textures()
sphmap_filename = read_viz_textures("clouds.jpg")

tex = vtk.vtkTexture()
imgReader = vtk.vtkJPEGReader()
imgReader.SetFileName(sphmap_filename)
tex.SetInputConnection(imgReader.GetOutputPort())
dragon.SetTexture(tex)

vertex_dec = "out vec3 TexCoords;"
vertex_impl = \
    """
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    TexCoords.xyz = reflect(vertexMC.xyz - camPos, normalize(normalMC));
    """
fragment_dec = \
    """
    uniform sampler2D texture_0;
    in vec3 TexCoords;
    """

fragment_impl = \
    """
    float phix = length(vec2(TexCoords.x, TexCoords.z));
    vec3 skyColor = texture(texture_0,
                    vec2(0.5*atan(TexCoords.z, TexCoords.x)/3.1415927
                    + 0.5, atan(TexCoords.y,phix)/3.1415927 + 0.5)).xyz;

    gl_FragData[0] = vec4(ambientColor + diffuse + specular +
                     specularColor*skyColor, opacity);
    """

shader_to_actor(dragon, "vertex", block="position", impl_code=vertex_impl,
                decl_code=vertex_dec)
shader_to_actor(dragon, "fragment", block="light", impl_code=fragment_impl,
                decl_code=fragment_dec)

world = vtk.vtkSkybox()
world.SetProjectionToSphere()
world.SetTexture(tex)
scene.add(world)

showm = window.ShowManager(scene)

showm.initialize()
showm.start()
