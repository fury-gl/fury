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


def get_actor_from_obj(filename):
    obj = read_viz_models(filename)
    obj = io.load_polydata(obj)

    norms = utils.get_polydata_normals(obj)
    obj = utils.set_polydata_normals(obj, norms)

    obj = utils.get_polymapper_from_polydata(obj)
    objactor = utils.get_actor_from_polymapper(obj)
    return objactor


dragon_actor = get_actor_from_obj("dragon.obj")
suzanne_actor = get_actor_from_obj("suzanne.obj")
satellite_actor = get_actor_from_obj("satellite_obj.obj")

satellite_actor.SetScale(0.05, 0.05, 0.05)
actor = dragon_actor

fetch_viz_textures()
sphmap_filename = read_viz_textures("clouds.jpg")


def set_toon(act):
    act.GetProperty().SetDiffuse(0.7)
    act.SetTexture(None)
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
    act.SetTexture(None)
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
exm = [dragon_actor, suzanne_actor, satellite_actor]

###############################################################################
# Create ListBox with the values as parameter.

models_values = ["Dragon", "Suzanne", "Satellite"]
shaders_values = ["Toon", "Reflect", "Gooch"]

panel = ui.Panel2D(size=(300, 450), color=(1, 1, 1), align="right")
panel.center = (150, 400)

combo_box = ui.ComboBox2D(items=shaders_values,
                          placeholder="Choose Shader", size=(250, 150))

listbox = ui.ListBox2D(values=models_values, size=(200, 150),
                       multiselection=False)

diff_tb = ui.TextBlock2D(text="Diffuse Value")
diffuse_slider = ui.LineSlider2D(center=(400, 230), initial_value=1.0,
                                 orientation='horizontal',
                                 min_value=0.0, max_value=1.0,
                                 text_alignment='top')


panel.add_element(listbox, (30, 290))
panel.add_element(combo_box, (30, 100))
panel.add_element(diff_tb, (30, 100))
panel.add_element(diffuse_slider, (30, 50))
###############################################################################
# Hide these text blocks for now


def hide_all_examples():
    for element in exm:
        element.SetVisibility(False)

hide_all_examples()

###############################################################################
# Function to render selected shader.


def shade_element(combobox):
    element = exx[shaders_values.index(combobox.selected_text)]
    element(actor)


def set_actor():
    global actor
    hide_all_examples()
    actor = exm[models_values.index(listbox.selected[0])]
    actor.SetVisibility(1)
    print(actor)


def set_diffuse(slider):
    value = slider.value
    actor.GetProperty().SetDiffuse(value)


combo_box.on_change = shade_element
listbox.on_change = set_actor
diffuse_slider.on_change = set_diffuse

###############################################################################
# Show Manager
#
# Now that all the elements have been initialised, we add them to the show
# manager.

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size,
                                  title="Shader Examples using UI")


show_manager.scene.add(dragon_actor)
show_manager.scene.add(suzanne_actor)
show_manager.scene.add(satellite_actor)
show_manager.scene.add(panel)


interactive = True

if interactive:
    show_manager.start()
