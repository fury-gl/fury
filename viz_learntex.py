import numpy as np
from fury import window, actor, ui, io, utils
from fury.data.fetcher import fetch_viz_models, read_viz_models
import vtk

#model = read_viz_models('utah.obj')
dragon = io.load_polydata('dragon.obj')
dragon = utils.get_polymapper_from_polydata(dragon)
dragon = utils.get_actor_from_polymapper(dragon)


tex = vtk.vtkTexture()
imgReader = vtk.vtkJPEGReader()
imgReader.SetFileName('sugar.jpg')
tex.SetInputConnection(imgReader.GetOutputPort())
dragon.SetTexture(tex)

mapper = dragon.GetMapper()

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
    TexCoords.xyz = reflect(vertexMC.xyz - camPos, normalize(normalMC));
    //TexCoords.xyz = normalMC;
    //TexCoords.xyz = vertexMC.xyz;
    """,
    False  # only do it once
)

mapper.SetFragmentShaderCode(
    """
    //VTK::System::Dec  // always start with this line
    //VTK::Output::Dec  // always have this line in your FS
    in vec3 TexCoords;
    uniform sampler2D texture_0;
    uniform vec2 tcoordMC;
    void main() {
        gl_FragData[0] = texture(texture_0, TexCoords.xy);
        //gl_FragData[0] = texture(texture_0, tcoordMC.xy);
    }
    """
)


scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(dragon)

window.show(scene)