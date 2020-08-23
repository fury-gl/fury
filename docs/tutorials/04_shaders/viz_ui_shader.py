"""
=====================
Using UI With Shaders
=====================

This example shows how to use the UI with shaders on an actor.

First, a bunch of imports.
"""

from fury import ui, window, io, utils
import vtk
from fury.data.fetcher import fetch_viz_models, read_viz_models,\
                              fetch_viz_textures, read_viz_textures

###############################################################################
# Let's start by downoading and loading the polydata of choice.
# For this example we use the stanford dragon.
# currently supported formats include OBJ, VKT, FIB, PLY, STL and XML

fetch_viz_models()
dragon = read_viz_models('dragon.obj')
dragon = io.load_polydata(dragon)
dragon = utils.get_polymapper_from_polydata(dragon)
dragon = utils.get_actor_from_polymapper(dragon)

fetch_viz_textures()
sphmap_filename = read_viz_textures("clouds.jpg")


def set_toon(act):
    act.GetProperty().SetDiffuse(0.7)
    mapper = act.GetMapper()
    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::Light::Impl',
        True,
        """
        //VTK::Light::Impl

        vec4 color;
        float ll = length(diffuse);
        if (ll > 0.95)
            color = vec4(1.0,0.5,0.5,1.0);
        else if (ll > 0.5)
            color = vec4(0.6,0.3,0.3,1.0);
        else if (ll > 0.25)
            color = vec4(0.4,0.2,0.2,1.0);
        else
            color = vec4(0.2,0.1,0.1,1.0);

        fragOutput0 = color;
        """,
        False
    )


def set_reflect(act):
    act.GetProperty().SetDiffuse(0.1)
    act.GetProperty().SetSpecular(0.8)
    act.GetProperty().SetSpecularPower(20)
    tex = vtk.vtkTexture()
    imgReader = vtk.vtkJPEGReader()
    imgReader.SetFileName(sphmap_filename)
    tex.SetInputConnection(imgReader.GetOutputPort())
    act.SetTexture(tex)

    mapper = act.GetMapper()

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::PositionVC::Dec",  # replace the normal block
        True,  # before the standard replacements
        """
        //VTK::PositionVC::Dec  // we still want the default
        out vec3 TexCoords;
        """,
        False  # only do it once
        )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::PositionVC::Impl",
        True,
        """

        //VTK::PositionVC::Impl  // we still want the default
        vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
        TexCoords.xyz = reflect(vertexMC.xyz - camPos, normalize(normalMC));
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Dec",
        True,
        """
        //VTK::Light::Dec

        uniform sampler2D texture_0;
        in vec3 TexCoords;
        """,
        False
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Impl",
        True,
        """
        //VTK::Light::Impl
        float phix = length(vec2(TexCoords.x, TexCoords.z));
        vec3 skyColor = texture(texture_0,
                        vec2(0.5*atan(TexCoords.z, TexCoords.x)/3.1415927+
                        0.5, atan(TexCoords.y,phix)/3.1415927 + 0.5)).xyz;
        gl_FragData[0] = vec4(ambientColor + diffuse + specular +
                         specularColor*skyColor, opacity);

        """,
        False
    )


def set_gooch(act):
    act.GetProperty().SetDiffuse(1.0)
    mapper = act.GetMapper()

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::ValuePass::Dec",
        True,
        """
        //VTK::ValuePass::Dec
        uniform float intensity;
        uniform float ambient;
        uniform float diffuse;
        vec3 CoolColor = vec3(0, 0, 1.0);
        float DiffuseCool = 0.25;
        float DiffuseWarm = 0.25;
        vec3 LightPosition = vec3(0, 10, 4);
        vec3 SurfaceColor = vec3(1, 0.75, 0.75);
        vec3 WarmColor = vec3(1.0, 1.0, 0);
        """,
        False
        )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::Light::Impl',
        True,
        """
        //VTK::Light::Impl

        vec3 kcool    = min(CoolColor + DiffuseCool * SurfaceColor, 1.0);
        vec3 kwarm    = min(WarmColor + DiffuseWarm * SurfaceColor, 1.0);
        vec3 kfinal   = mix(kcool, kwarm, diffuse);
        fragOutput0 = vec4(min(kfinal + specular, 1.0), 1.0);
        """,
        False
        )

exx = [set_toon, set_reflect, set_gooch]

###############################################################################
# Create ListBox with the values as parameter.

values = ["Toon", "Reflect", "Gooch"]
listbox = ui.ListBox2D(
    values=values, position=(10, 300), size=(200, 200), multiselection=False
)

###############################################################################
# Function to render selected shader.


def shade_element():
    element = exx[values.index(listbox.selected[0])]
    element(dragon)

listbox.on_change = shade_element

###############################################################################
# Show Manager
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="Shader Examples using UI")

show_manager.scene.add(listbox)
show_manager.scene.add(dragon)

interactive = False

if interactive:
    show_manager.start()
