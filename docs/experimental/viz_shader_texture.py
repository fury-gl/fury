from fury import window
from viz_shader_canvas import cube


import vtk


scene = window.Scene()

canvas_actor = cube()
canvas_actor.GetProperty().BackfaceCullingOff()
scene.add(canvas_actor)
mapper = canvas_actor.GetMapper()

texture = vtk.vtkTexture()
texture.CubeMapOn()
file = "sugar.jpg"

imgReader = vtk.vtkJPEGReader()
imgReader.SetFileName(file)

for i in range(6):
    flip = vtk.vtkImageFlip()
    flip.SetInputConnection(imgReader.GetOutputPort())
    flip.SetFilteredAxis(1)
    texture.SetInputConnection(i, flip.GetOutputPort())

canvas_actor.SetTexture(texture)

# // Add new code in default VTK vertex shader
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
    "//VTK::PositionVC::Impl",  # replace the normal block
    True,  # before the standard replacements
    """
    //VTK::PositionVC::Impl  // we still want the default
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    //TexCoords.xyz = normalMC;
    TexCoords.xyz = vertexMC.xyz;
    """,
    False  # only do it once
)

mapper.SetFragmentShaderCode(
    """
    //VTK::System::Dec  // always start with this line
    //VTK::Output::Dec  // always have this line in your FS
    in vec3 TexCoords;
    uniform samplerCube texture_0;
    
    void main() {
        gl_FragData[0] = texture(texture_0, TexCoords);
    }
    """
)

window.show(scene)
