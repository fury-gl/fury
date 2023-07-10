import numpy as np
from fury import window, actor, shaders
import vtk
import os
from fury.shaders import compose_shader, import_fury_shader
# Function to capture the framebuffer and bind it as a texture to the billboard
def capture_and_bind_texture(source_actor : actor.Actor, scene : window.Scene, target_actor : actor.Actor = None):
    # Capture the framebuffer
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(scene.GetRenderWindow())
    
    windowToImageFilter.Update()

    # Convert the framebuffer to a texture
 
    texture = vtk.vtkTexture()
    texture.SetInputConnection(windowToImageFilter.GetOutputPort())
    texture.SetEdgeClamp(texture.ClampToBorder)
    texture.SetBorderColor(0.0, 0.0, 1.0, 1.0)
    texture.SetWrap(3)
    texture.MipmapOn()
    texture.SetBlendingMode(0) # Additive Blend

    # Bind the framebuffer texture to the billboard
    if target_actor == None:
        source_actor.SetTexture(texture)
    else:
        target_actor.SetTexture(texture)



def shader_custom_uniforms(actor : actor.Actor, shader_type : str):
    """Eases the passing of uniform values to the shaders by returning ``actor.GetShaderProperty().GetVertexCustomUniforms()``, 
    that give access to the ``SetUniform`` methods.
    Parameters
    ----------
    actor : actor.Actor
          Actor which the uniform values will be passed to.
    shader_type : str
          Shader type of the uniform values to be passed. It can be: 
          * "vertex"
          * "fragment"
          * "geometry"
    """
    if shader_type == "vertex":
        return actor.GetShaderProperty().GetVertexCustomUniforms()
    elif shader_type == "fragment":
        return actor.GetShaderProperty().GetFragmentCustomUniforms()
    elif shader_type == "geometry":
        return actor.GetShaderProperty().GetGeometryCustomUniforms()
    else: raise ValueError("Shader type unknown.")


def normalize(img : np.ndarray, min, max):
  """Function that converts an image to the given desired range."""
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(max - min) + min


frag_dec = """
uniform sampler2D screenTexture;
"""

kde_dec = """
float kde(vec3 point, vec3 coord, float sigma){
    return exp(-1.0*pow(length(point - coord), 2.0)/(2.0*sigma*sigma) );
} 
"""

kde_impl = """
vec3 renorm_tex = normalizedVertexMCVSOutput*0.5 + 0.5;
vec3 last_kde = texture(screenTexture, renorm_tex.xy).rgb;

if(i == 0) last_kde = vec3(0.0);

float current_kde = kde(p, normalizedVertexMCVSOutput, sigma);
//color = vec3(current_kde) + last_kde/n_points;
color = 2.0*mix(vec3(current_kde), last_kde, 0.5);
//color = vec3(renorm_tex.xy, 0.0);
"""


tex_dec ="""
uniform sampler2D screenTexture;
"""

tex_impl ="""
vec3 renorm_tex = normalizedVertexMCVSOutput*0.5 + 0.5;
color = texture(screenTexture, renorm_tex.xy).rgb;
"""

fragoutput = """
fragOutput0 = vec4(color, 1.0);
"""


fs_dec = shaders.compose_shader([frag_dec, kde_dec])

fs_impl = shaders.compose_shader([kde_impl, fragoutput])

tex_impl = shaders.compose_shader([tex_impl, fragoutput])

width, height = (600, 600)

scene = window.Scene()

scene.set_camera(position=(-6, 5, -10),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 0.0))

scene.DebugOn()
scene.GlobalWarningDisplayOn()

manager = window.ShowManager(
    scene,
    "demo",
    (width,
     height),
    reset_camera=True,
    order_transparent=True)

# manager.window.SetOffScreenRendering(True)

manager.scene.GetRenderWindow().DebugOn()
manager.scene.GetRenderWindow().GlobalWarningDisplayOn()


scale = np.array([[3.4, 3.4, 0.0]])


billboard = actor.billboard(np.array([[0.0, 0.0, 0.0]]), (0.0, 0.0, 1.0), scales=scale, fs_dec=fs_dec, fs_impl=fs_impl)
textured_billboard = actor.billboard(np.array([[0.0, 0.0, 0.0]]), (0.0, 0.0, 1.0), scales=scale, fs_dec=tex_dec, fs_impl=tex_impl)
# shaders.shader_apply_effects(manager.window, billboard, window.gl_disable_depth)
# shaders.shader_apply_effects(manager.window, billboard, window.gl_set_additive_blending)


manager.scene.add(billboard)
manager.initialize()


n_points = 20
sigma = 0.22
shader_custom_uniforms(billboard, "fragment").SetUniformf("sigma", sigma)
shader_custom_uniforms(billboard, "fragment").SetUniformf("n_points", float(n_points))
shader_custom_uniforms(billboard, "fragment").SetUniform2f("res", [width, height])



points = np.random.rand(n_points, 3)
points = normalize(points, -1, 1)
# print(points)
# points = np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.7, 0.7, 0.0]])


# Render the scene and capture the framebuffer after each mixing step
manager.window.SetOffScreenRendering(True)
for i in range(0, n_points - 1): 
    shader_custom_uniforms(billboard, "fragment").SetUniformi("i", i)
    shader_custom_uniforms(billboard, "fragment").SetUniform3f("p", points[i, :].tolist())
    
    manager.render()

    capture_and_bind_texture(billboard, manager.scene)

shader_custom_uniforms(billboard, "fragment").SetUniformi("i", i)
shader_custom_uniforms(billboard, "fragment").SetUniform3f("p", points[n_points - 1, :].tolist())
    
manager.render()

manager.window.SetOffScreenRendering(False)

capture_and_bind_texture(billboard, manager.scene, textured_billboard)



# manager.scene.RemoveActor(billboard)
# manager.scene.add(textured_billboard)

manager.start()