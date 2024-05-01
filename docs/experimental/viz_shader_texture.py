import numpy as np
from viz_shader_canvas import cube
import vtk

from fury import window
from fury.utils import rgb_to_vtk

scene = window.Scene()

selected_actor = 'sphere'

if selected_actor == 'cube':
    canvas_actor = cube()

if selected_actor == 'sphere':
    # Generate an sphere polydata
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(300)
    sphere.SetPhiResolution(300)

    norms = vtk.vtkPolyDataNormals()
    norms.SetInputConnection(sphere.GetOutputPort())

    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputConnection(norms.GetOutputPort())

    canvas_actor = vtk.vtkActor()
    canvas_actor.SetMapper(mapper)

texture = vtk.vtkTexture()
texture.CubeMapOn()

selected_texture = 'numpy'

if selected_texture == 'file':
    file = 'sugar.jpg'
    imgReader = vtk.vtkJPEGReader()
    imgReader.SetFileName(file)
    for i in range(6):
        flip = vtk.vtkImageFlip()
        flip.SetInputConnection(imgReader.GetOutputPort())
        flip.SetFilteredAxis(1)
        texture.SetInputConnection(i, flip.GetOutputPort())

if selected_texture == 'numpy':
    arr = 255 * np.random.randn(512, 512, 3)
    arr[:256] = np.array([255, 0, 0])
    grid = rgb_to_vtk(arr.astype(np.uint8))
    for i in range(6):
        texture.SetInputDataObject(i, grid)

canvas_actor.SetTexture(texture)
mapper = canvas_actor.GetMapper()

# // Add new code in default VTK vertex shader
mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::PositionVC::Dec',  # replace the normal block
    True,  # before the standard replacements
    """
    //VTK::PositionVC::Dec  // we still want the default
    out vec3 TexCoords;
    """,
    False,  # only do it once
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    '//VTK::PositionVC::Impl',  # replace the normal block
    True,  # before the standard replacements
    """
    //VTK::PositionVC::Impl  // we still want the default
    vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    TexCoords.xyz = reflect(vertexMC.xyz - camPos, normalize(normalMC));
    //TexCoords.xyz = normalMC;
    //TexCoords.xyz = vertexMC.xyz;
    """,
    False,  # only do it once
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

scene.add(canvas_actor)
# scene.add(actor.axes())
# scene.add(texture(np.random.randn(512, 512, 3)))

window.show(scene)
