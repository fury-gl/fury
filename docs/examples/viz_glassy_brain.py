from dipy.data import get_fnames
from fury import actor, ui, window
from fury.data import read_viz_textures
from fury.io import load_polydata
from fury.utils import get_actor_from_polydata
from fury.shaders import shader_to_actor


import numpy as np
import os
import vtk


__FRAGMENT_DEC = \
    """
    float chiGGX(float v)
    {
        return v > 0 ? 1. : .0;
    }
    
    vec3 FresnelSchlick(float HdV, vec3 F0)
    {
        return F0 + (1 - F0) * pow(1 - HdV, 5);
    }
    
    float GGXDistribution(float NdH, float alpha)
    {
        float alpha2 = alpha * alpha;
        float NdH2 = NdH * NdH;
        float den = NdH2 * alpha2 + (1 - NdH2);
        return (chiGGX(NdH) * alpha2) / (PI * den * den);
    }
    
    float GGXPartialGeometryTerm(float VdH, float VdN, float alpha)
    {
        float cVdH = clamp(VdH, .0, 1.);
        float chi = chiGGX(cVdH / clamp(VdN, .0, 1.));
        float tan2 = (1 - cVdH) / cVdH;
        return (chi * 2) / (1 + sqrt(1 + alpha * alpha * tan2));
    }
    """
__FRAGMENT_IMPL = \
    """
    vec3 ior = vec3(2.4);
    vec3 F0_t = abs((1. - ior) / (1. + ior));
    F0_t *= F0_t;
    F0_t = mix(F0_t, albedo, metallic);
    
    vec3 Lo_t = vec3(.0);
    
    vec3 F_t = F_Schlick(1., F0_t);
    vec3 specular_t = D * Vis * F_t;
    vec3 diffuse_t = (1. - metallic) * (1. - F_t) * DiffuseLambert(albedo);
    Lo_t += (diffuse_t + specular_t) * lightColor0 * NdV;
    
    vec3 kS_t = F_SchlickRoughness(max(NdV, .0), F0_t, roughness);
    vec3 kD_t = 1. - kS_t;
    kD_t *= 1. - metallic;
    vec3 ambient_t = (kD_t * irradiance * albedo + prefilteredColor * (kS_t * brdf.r + brdf.g));
    vec3 color_t = ambient_t + Lo_t;
    color_t = mix(color_t, color_t * ao, aoStrengthUniform);
    color_t += emissiveColor;
    color_t = pow(color_t, vec3(1. / 2.2));
    fragOutput0 = vec4(color_t, opacity);
    """


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
    global obj_actor
    obj_actor.GetProperty().SetMetallic(slider._value)


def change_slice_roughness(slider):
    global obj_actor
    obj_actor.GetProperty().SetRoughness(slider._value)


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
    return get_actor_from_polydata(polydata)


def obj_spheres(radii=2, theta=32, phi=32):
    centers = [[-5, 5, 0], [0, 5, 0], [5, 5, 0], [-5, 0, 0], [0, 0, 0],
               [5, 0, 0], [-5, -5, 0], [0, -5, 0], [5, -5, 0]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0],
              [0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    return actor.sphere(centers, colors, radii=radii, theta=theta, phi=phi)


def win_callback(obj, event):
    global panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        panel.re_align(size_change)


if __name__ == '__main__':
    global panel, size

    #obj_actor = obj_brain()
    obj_actor = obj_spheres()

    metallic = 0
    roughness = 0

    obj_actor.GetProperty().SetInterpolationToPBR()
    obj_actor.GetProperty().SetMetallic(metallic)
    obj_actor.GetProperty().SetRoughness(roughness)

    #shader_to_actor(obj_actor, 'vertex', debug=True)
    shader_to_actor(obj_actor, "fragment", decl_code=__FRAGMENT_DEC)
    shader_to_actor(obj_actor, "fragment", impl_code=__FRAGMENT_IMPL,
                    block="light", debug=False)

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

    panel = ui.Panel2D((320, 480), position=(-25, 5), color=(.25, .25, .25),
                       opacity=.75, align='right')

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

    label_pad_x = .02

    panel.add_element(slider_label_subsurface, (label_pad_x, .95))
    panel.add_element(slider_label_metallic, (label_pad_x, .85))
    panel.add_element(slider_label_specular, (label_pad_x, .75))
    panel.add_element(slider_label_specular_tint, (label_pad_x, .65))
    panel.add_element(slider_label_roughness, (label_pad_x, .55))
    panel.add_element(slider_label_anisotropic, (label_pad_x, .45))
    panel.add_element(slider_label_sheen, (label_pad_x, .35))
    panel.add_element(slider_label_sheen_tint, (label_pad_x, .25))
    panel.add_element(slider_label_clearcoat, (label_pad_x, .15))
    panel.add_element(slider_label_clearcoat_gloss, (label_pad_x, .05))

    length = 160
    text_template = '{value:.1f}'

    slider_slice_subsurface = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_metallic = ui.LineSlider2D(
        initial_value=metallic, max_value=1, length=length,
        text_template=text_template)
    slider_slice_specular = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_specular_tint = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=roughness, max_value=1, length=length,
        text_template=text_template)
    slider_slice_anisotropic = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_sheen = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_sheen_tint = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_clearcoat = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)
    slider_slice_clearcoat_gloss = ui.LineSlider2D(
        initial_value=0, max_value=1, length=length,
        text_template=text_template)

    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_roughness.on_change = change_slice_roughness

    slice_pad_x = .42

    panel.add_element(slider_slice_subsurface, (slice_pad_x, .95))
    panel.add_element(slider_slice_metallic, (slice_pad_x, .85))
    panel.add_element(slider_slice_specular, (slice_pad_x, .75))
    panel.add_element(slider_slice_specular_tint, (slice_pad_x, .65))
    panel.add_element(slider_slice_roughness, (slice_pad_x, .55))
    panel.add_element(slider_slice_anisotropic, (slice_pad_x, .45))
    panel.add_element(slider_slice_sheen, (slice_pad_x, .35))
    panel.add_element(slider_slice_sheen_tint, (slice_pad_x, .25))
    panel.add_element(slider_slice_clearcoat, (slice_pad_x, .15))
    panel.add_element(slider_slice_clearcoat_gloss, (slice_pad_x, .05))

    scene.add(panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
