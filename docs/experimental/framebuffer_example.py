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


def texture_to_actor(actor: lib.Actor, location : str, texture : np.array, texture_channel : int):
    """WIP function to pass custom texture info to an actor"""

    width, height, n = texture.shape

    # Setting the image data
    info = vtk.vtkInformation()
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(width, height, 1)
    imageData.SetNumberOfScalarComponents(n, info)
    imageData.AllocateScalars(vtk.VTK_FLOAT, 4)

    for y in range(width):
        for x in range(height):
            pixel = texture[y, x]
            for i in range(n):
                imageData.SetScalarComponentFromFloat(x, y, 0, i, pixel[i])



    # Setting the texture
    texture_obj = lib.Texture()
    texture_obj.SetInputDataObject(imageData)
    texture_obj.InterpolateOn()
    texture_obj.Update()

    actor.SetTexture(texture_obj)
    actor.GetProperty().SetTexture(location, texture_obj)




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

billboard_frag =   """#ifdef GL_FRAGMENT_PRECISION_MEDIUM 
                        precision mediump float;
                    #else
                        precision lowp float;
                        precision lowp int;
                    #endif
                    
                    varying vec2 out_tex;
                    
                    uniform float t;
                    uniform vec2 res;
                    uniform sampler2D textureSampler;
                    
                    
                    void main(){
                    
                        vec3 color = vec3(gl_FragCoord.xy/res, 1.0);
                        vec4 texture = texture(textureSampler, out_tex);
                    
                        // gl_FragColor = vec4(color, 1.0);\
                        gl_FragColor = vec4(exp(-color.x*color.x), exp(-color.y*color.y), exp(-color.z*color.z), 1.0);\
                        // gl_FragColor = vec4(abs(sin(t)), abs(cos(t)), 1.0, 1.0);\
                    
                    
                    }"""

vert_decl = """/* Billboard  vertex shader declaration */ 
            in vec3 center; 
            in vec2 in_tex; 
            out vec3 centerVertexMCVSOutput; 
            out vec3 normalizedVertexMCVSOutput; 
            varying vec2 out_tex;"""

vert_impl = """/* Billboard  vertex shader implementation */ 
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


frag = """#ifdef GL_FRAGMENT_PRECISION_MEDIUM 
    precision mediump float; 
#else 
    precision lowp float; 
    precision lowp int; 
#endif 

varying vec2 out_tex; 
uniform float t; 
uniform vec2 res; 
layout(binding = 0) uniform sampler2D textureSampler; 
uniform vec3 point; 

float kde(vec3 point, vec3 coord, float sigma){ 
    return exp(-pow(length(abs(point - coord)), 2.0)/(2.0*sigma*sigma)); 
} 

void main(){ 
    
    vec3 color = vec3(gl_FragCoord.xy/res, 1.0); 
    vec3 texture = texture(textureSampler, out_tex); 
    vec3 k = vec3(1.0, kde(point, vec3(gl_FragCoord.xy/res, 0.0), 10.0), 1.0); 
    
    // gl_FragColor = vec4(color, 1.0); 
    gl_FragColor = vec4((k + texture)*0.5, 1.0); 
    // gl_FragColor = vec4(abs(sin(t)), abs(cos(t)), 1.0, 1.0);}"""


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
actor.shader_to_actor(billboard, "vertex", billboard_vert_impl, billboard_vert_decl)
actor.replace_shader_in_actor(billboard, "fragment", billboard_frag)
shader_custom_uniforms(billboard, "fragment").SetUniform2f("res", [width, height])

# billboard_tex = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
billboard_tex = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
actor.attribute_to_actor(billboard, billboard_tex, "in_tex")

scene.GetRenderWindow().DebugOn()
scene.GetRenderWindow().GlobalWarningDisplayOn()

# Actor adding
scene.add(billboard)





# Points setup
n_points = 10
points = np.random.rand(10, 3)


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
manager.initialize()
# manager.start()

FBO.DebugOn()
FBO.GlobalWarningDisplayOn()

FBO.SetContext(manager.window) # Sets the context for the FBO. 
FBO.PopulateFramebuffer(width, height, True, 1, vtk.VTK_UNSIGNED_CHAR, False, 24, 0)

# Checking FBO status
print("FBO of index:", FBO.GetFBOIndex()) 
print("Number of color attachments:", FBO.GetNumberOfColorAttachments())


# Below, how to render things with a FBO according to VTK's website
# FBO.SaveCurrentBindingsAndBuffers()
# FBO.Bind()
# FBO.ActivateBuffer(0)
# scene.Clear()
# FBO.Start(width, height)
# # Render
# FBO.RestorePreviousBindingsAndBuffers()
