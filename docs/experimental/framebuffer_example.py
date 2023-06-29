import numpy as np
from fury import window, actor, lib, shaders
import vtk


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



# Don't mind these shaders, I just set them up to focus first on the framebuffer, then after on the shaders themselves
billboard_vert_decl =  """/* Billboard  vertex shader declaration */
in vec3 center;
in vec2 in_tex;

out vec3 centerVertexMCVSOutput;
out vec3 normalizedVertexMCVSOutput;
varying vec2 out_tex;"""

billboard_vert_impl =  """/* Billboard  vertex shader implementation */
centerVertexMCVSOutput = center;
normalizedVertexMCVSOutput = vertexMC.xyz - center; // 1st Norm. [-scale, scale]
float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
float size = abs(normalizedVertexMCVSOutput.x) * 2;
normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1]
vec2 billboardSize = vec2(size, size); // Fixes the scaling issue
vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPositionMC = center +
    cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x +
    cameraUpMC * billboardSize.y * normalizedVertexMCVSOutput.y;
out_tex = in_tex;
gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);"""


frag_decl = """in vec3 normalizedVertexMCVSOutput;
"""

kde_func = """float kde(vec3 point, vec3 coord, float sigma){ 
    return exp(-pow(length(abs(point - coord)), 2.0)/(2.0*sigma*sigma)); 
}; """

frag_impl = """
vec3 k = vec3(kde(point0, vec3(gl_FragCoord.xy/res0, 0.0), 10.0)); 
fragOutput0 = vec4(k, 0.5);"""


frag = """

varying vec2 out_tex;
void main(){ 
    gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0); 
}"""


frag_2 = """

varying vec2 out_tex;
                    
uniform sampler2D screenTexture;

void main(){ 
    vec4 texture = texture(screenTexture, out_tex);
    gl_FragColor = texture; 
    //gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); 
}"""


width, height = (1920, 1080)

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
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)


scale = np.array([[width, height, 0.0]])

# Actor setup
billboard = actor.billboard(np.array([[0.0, 0.0, 0.0]]), (1.0, 0.0, 0.0), scales=scale)

# Adding first the kde rendering shaders
actor.shader_to_actor(billboard, "vertex", billboard_vert_impl, billboard_vert_decl)
actor.replace_shader_in_actor(billboard, "fragment", frag)
billboard_tex = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
actor.attribute_to_actor(billboard, billboard_tex, "in_tex")



# Actor adding
manager.scene.add(billboard)



manager.scene.GetRenderWindow().DebugOn()
manager.scene.GetRenderWindow().GlobalWarningDisplayOn()



# FBO setup

# Useful links
# https://vtk.org/doc/nightly/html/classvtkOpenGLFramebufferObject.html#details
# https://gitlab.kitware.com/vtk/vtk/-/blob/v9.2.0/Rendering/OpenGL2/Testing/Cxx/TestWindowBlits.cxx
# https://vtk.org/doc/nightly/html/classvtkTextureObject.html
# https://vtk.org/doc/nightly/html/classvtkOpenGLRenderWindow.html
# https://learnopengl.com/Advanced-OpenGL/Framebuffers
# https://github.com/JoaoDell/Comp-Grafica/blob/main/shadow3d.c
# https://github.com/Kitware/VTK/blob/8f88edb91d2efea2d9cef1a1399d7a856c47f3be/Rendering/OpenGL2/vtkOpenGLFramebufferObject.cxx 
# https://fury.gl/latest/auto_examples/04_demos/viz_fine_tuning_gl_context.html#sphx-glr-auto-examples-04-demos-viz-fine-tuning-gl-context-py



FBO = vtk.vtkOpenGLFramebufferObject()

manager.window.SetOffScreenRendering(True)
manager.initialize() # sets everything for rendering

FBO.DebugOn()
FBO.GlobalWarningDisplayOn()

FBO.SetContext(manager.window) # Sets the context for the FBO. 
FBO.SaveCurrentBindingsAndBuffers()
FBO.PopulateFramebuffer(width, height, True, 1, vtk.VTK_UNSIGNED_CHAR, False, 24, 0)
FBO.RestorePreviousBindingsAndBuffers()





# Checking FBO status
print("FBO of index:", FBO.GetFBOIndex()) 
print("Number of color attachments:", FBO.GetNumberOfColorAttachments())
FBO.RestorePreviousBindingsAndBuffers()

# RENDER TIME
shaders.shader_apply_effects(manager.window, billboard, window.gl_disable_depth)
shaders.shader_apply_effects(manager.window, billboard, window.gl_set_normal_blending)

FBO.SaveCurrentBindingsAndBuffers()
FBO.Bind()
FBO.ActivateBuffer(0)

manager.render()

color_texture = FBO.GetColorAttachmentAsTextureObject(0)

print("Color texture")
print(color_texture)


print("Color texture handle:", color_texture.GetHandle()) # corresponds to the value generated when we call glGenTextures()
print("Color texture unit:", color_texture.GetTextureUnit()) # corresponds to the texture unit the texture is currently bound (GL_TEXTURE0, etc)
print("Color texture target:", color_texture.GetTarget())

FBO.RestorePreviousBindingsAndBuffers()


# WIP below
manager.window.SetOffScreenRendering(False) # Then, after rendering everything in the offscreen FBO, time to render it to a simple billboard
actor.replace_shader_in_actor(billboard, "fragment", frag_2)
shaders.shader_apply_effects(manager.window, billboard, window.gl_disable_blend)

color_texture.Activate()

interactive = True

if interactive:
    manager.start()
