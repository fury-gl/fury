//VTK::System::Dec
//VTK::PositionVC::Dec
uniform float linewidth;
uniform mat4 MCDCMatrix;
uniform mat4 MCVCMatrix;
uniform mat4 MCWCMatrix;
uniform mat4 VCMCMatrix;
uniform float scale;

//VTK::PrimID::Dec

//in vec4 vertexColorVSOutput[];
//out vec4 vertexColorGSOutput;

//VTK::Color::Dec
//VTK::Normal::Dec
//VTK::Light::Dec
//VTK::TCoord::Dec
//VTK::Picking::Dec
//VTK::DepthPeeling::Dec
//VTK::Clip::Dec
//VTK::Output::Dec

// convert points to triangle strips
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

void main() {



    const vec2 coordinates [] = vec2 [] (vec2 (-0.5f, -0.5f),
                                   vec2 (0.5f, -0.5f),
                                   vec2 (-0.5f, 0.5f),
                                   vec2 (0.5f, 0.5f)
                                   );

    for (int j = 0; j < 4; j++) {
        gl_Position = MCDCMatrix * ( (gl_in [0].gl_Position ) +  VCMCMatrix * vec4(coordinates[j], 0, 0));
        vertexColorGSOutput = vertexColorVSOutput[0];
        EmitVertex ();
        }
    EndPrimitive();




}
