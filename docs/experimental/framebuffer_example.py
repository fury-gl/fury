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

billboard_vert_decl =  "/* Billboard  vertex shader declaration */\
                        in vec3 center;\
                        in vec2 in_tex;\
                        \
                        out vec3 centerVertexMCVSOutput;\
                        out vec3 normalizedVertexMCVSOutput;\
                        varying vec2 out_tex;"

billboard_vert_impl =  "/* Billboard  vertex shader implementation */\
                        centerVertexMCVSOutput = center;\
                        normalizedVertexMCVSOutput = vertexMC.xyz - center; // 1st Norm. [-scale, scale]\
                        float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);\
                        float size = abs(normalizedVertexMCVSOutput.x) * 2;\
                        normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1]\
                        vec2 billboardSize = vec2(size, size); // Fixes the scaling issue\
                        vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);\
                        vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);\
                        vec3 vertexPositionMC = center +\
                            cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x +\
                            cameraUpMC * billboardSize.y * normalizedVertexMCVSOutput.y;\
                        out_tex = in_tex;\
                        gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);"

billboard_frag =   "#ifdef GL_FRAGMENT_PRECISION_MEDIUM \
                        precision mediump float;\
                    #else\
                        precision lowp float;\
                        precision lowp int;\
                    #endif\
                    \
                    varying vec2 out_tex;\
                    \
                    uniform float t;\
                    uniform vec2 res;\
                    uniform sampler2D textureSampler;\
                    \
                    \
                    void main(){\
                    \
                        vec3 color = vec3(gl_FragCoord.xy/res, 1.0);\
                        vec4 texture = texture(textureSampler, out_tex);\
                    \
                        // gl_FragColor = vec4(color, 1.0);\
                        gl_FragColor = vec4(exp(-color.x*color.x), exp(-color.y*color.y), exp(-color.z*color.z), 1.0);\
                        // gl_FragColor = vec4(abs(sin(t)), abs(cos(t)), 1.0, 1.0);\
                    \
                    \
                    }"

vert_decl = "/* Billboard  vertex shader declaration */ \
            in vec3 center; \
            in vec2 in_tex; \
            out vec3 centerVertexMCVSOutput; \
            out vec3 normalizedVertexMCVSOutput; \
            varying vec2 out_tex;"

vert_impl = "/* Billboard  vertex shader implementation */ \
            centerVertexMCVSOutput = center; \
            normalizedVertexMCVSOutput = vertexMC.xyz - center; // 1st Norm. [-scale, scale] \
            float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x); \
            float size = abs(normalizedVertexMCVSOutput.x) * 2; \
            normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1] \
            vec2 billboardSize = vec2(size, size); // Fixes the scaling issue \
            vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]); \
            vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]); \
            vec3 vertexPositionMC = center + \
                cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x + \
                cameraUpMC * billboardSize.y * normalizedVertexMCVSOutput.y; \
            out_tex = in_tex; \
            gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);"


frag = "#ifdef GL_FRAGMENT_PRECISION_MEDIUM \
    precision mediump float; \
#else \
    precision lowp float; \
    precision lowp int; \
#endif \
\
varying vec2 out_tex; \
uniform float t; \
uniform vec2 res; \
layout(binding = 0) uniform sampler2D textureSampler; \
uniform vec3 point; \
\
float kde(vec3 point, vec3 coord, float sigma){ \
    return exp(-pow(length(abs(point - coord)), 2.0)/(2.0*sigma*sigma)); \
} \
\
void main(){ \
    \
    vec3 color = vec3(gl_FragCoord.xy/res, 1.0); \
    vec3 texture = texture(textureSampler, out_tex); \
    vec3 k = vec3(1.0, kde(point, vec3(gl_FragCoord.xy/res, 0.0), 10.0), 1.0); \
    \
    // gl_FragColor = vec4(color, 1.0); \
    gl_FragColor = vec4((k + texture)*0.5, 1.0); \
    // gl_FragColor = vec4(abs(sin(t)), abs(cos(t)), 1.0, 1.0);}"


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




color_texture = vtk.vtkTextureObject()
color_texture.SetContext(scene.GetRenderWindow())
color_texture.Bind()
color_texture.SetDataType(vtk.VTK_UNSIGNED_CHAR)
color_texture.SetInternalFormat(vtk.VTK_RGB)
color_texture.SetFormat(vtk.VTK_RGB)
color_texture.SetMinificationFilter(0)
color_texture.SetMagnificationFilter(0)
try:
    color_texture.Allocate2D(width, height, 3, vtk.VTK_UNSIGNED_CHAR)
    print("Color texture allocation successful.")
except:
    raise RuntimeError("Color texture allocation failed.")


depth_texture = vtk.vtkTextureObject()
depth_texture.SetMinificationFilter(0)
depth_texture.SetMagnificationFilter(0)
depth_texture.SetContext(scene.GetRenderWindow())
try:
    depth_texture.Allocate2D(width, height, 1, vtk.VTK_UNSIGNED_CHAR)
    print("Depth texture allocation successful.")
except:
    raise RuntimeError("Depth texture allocation failed.")

FBO = vtk.vtkOpenGLFramebufferObject()
# FBO.AddObserver(0, FBO.GetCommand(0)) # This may help?
FBO.DebugOn()
FBO.GlobalWarningDisplayOn()
# print(scene.GetRenderWindow().SupportsOpenGL()) # FOR SOME REASON THIS FUNCTION HERE MAKES SOME WORK


# scene.GetRenderWindow().SetUseOffScreenBuffers(True)
# print(scene.GetRenderWindow())
print(FBO.IsSupported(scene.GetRenderWindow())) # the context supports the required extensions  
FBO.SetContext(scene.GetRenderWindow()) # Sets the context for the FBO.  >>>>>>THE PROBLEM IS HERE<<<<<<<


# print(FBO.CheckFrameBufferStatus(35648)) #This does not work (the code inside is the integer related to GL_FRAMEBUFFER)
# FBO.Start(width, height) # This is the only method found that actually outputs an error
FBO.SaveCurrentBindingsAndBuffers()
# FBO.PopulateFramebuffer(width, height, True, 1, vtk.VTK_UNSIGNED_CHAR, True, 24, 0) # This may replace the need to manually declare TextureObjects
FBO.AddColorAttachment(0, color_texture) # Attaches a color texture to this FBO
FBO.AddDepthAttachment(depth_texture) # Attaches a depth texture to this FBO
# FBO.Bind() # Binding method
print(FBO.GetFBOIndex()) # The problem can be verified here: it is outputting 0, so it means the FBO could not get generated
print(FBO.GetNumberOfColorAttachments())
FBO.RestorePreviousBindingsAndBuffers() # Restore to the previous bindings and buffers before that
# FBO.UnBind() # This is an important method because it sets the binding for the default FBO

# Important feature because with this the OpenGL states can be kept track of
# Some methods here include glEnable, glDisable, etc
state = FBO.GetContext().GetState()

print("Número máximo de Alvos Ativos: ", FBO.GetMaximumNumberOfActiveTargets())

manager = window.ShowManager(
    scene,
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)



# Below, how to render things with a FBO according to VTK's website
# FBO.SaveCurrentBindingsAndBuffers()
# FBO.Bind()
# FBO.ActivateBuffer(0)
# scene.Clear()
# FBO.Start(width, height)
# # Render
# FBO.RestorePreviousBindingsAndBuffers()

# Beginning of rendering
manager.initialize()
t = 0
while True:
    t += 0.005
    shader_custom_uniforms(billboard, "fragment").SetUniformf("t", t)
 
    # Lets the renderer know this object was modified
    billboard.Modified()
    

    # Render to the FBO
    manager.window.SetUseOffScreenBuffers(True)
    FBO.SaveCurrentBindingsAndBuffers()
    FBO.Bind()
    FBO.ActivateBuffer(0)
    FBO.ActivateBuffer(0)
    for i in range(n_points):
        actor.shader_to_actor(billboard, "vertex", vert_impl, vert_decl)
        actor.replace_shader_in_actor(billboard, "fragment", frag)
        shader_custom_uniforms(billboard, "fragment").SetUniform3f("point", points[i])
        FBO.AddColorAttachment(0, color_texture) # Attaches a color texture to this FBO
        FBO.Start(width, height)
        print(i)
        manager.render()
    
    FBO.RestorePreviousBindingsAndBuffers()
    manager.window.SetUseOffScreenBuffers(False)
    FBO.UnBind() 
    
    # Render

    # Render the scene
    manager.window.MakeCurrent()
    actor.shader_to_actor(billboard, "vertex", billboard_vert_impl, billboard_vert_decl)
    actor.replace_shader_in_actor(billboard, "fragment", billboard_frag)
    shader_custom_uniforms(billboard, "fragment").SetUniform2f("res", [width, height])
    manager.render()

    # Allow user interaction with the scene
    manager.iren.ProcessEvents()

    # Exit the loop if the user closes the window
    if manager.window.GetDesiredUpdateRate() <= 0.0:
        FBO.FastDelete()
        manager.window.Finalize()
        manager.exit()
        break