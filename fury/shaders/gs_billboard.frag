//VTK::System::Dec
//VTK::PositionVC::Dec
uniform mat4 MCDCMatrix;
uniform mat4 MCVCMatrix;

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

const mat4 coordinates = mat4 (vec4 (-0.5,-0.5, 0, 0),
                               vec4 ( 0.5,-0.5, 0, 0),
                               vec4 (-0.5, 0.5, 0, 0),
                               vec4 ( 0.5, 0.5, 0, 0));

void main() {

    // trick to multiply inv(MCVCMatrix) * all four offsets
    mat4 c =  coordinates ;

    // Adding the center to the offsets
    for (int j = 0; j < 4; j++) {
        c[j] += gl_in [0].gl_Position;
    }

    // transferring all four vertices into device coords
    mat4 vertices = MCDCMatrix *  c;

    // send same color for the whole billboard
    vertexColorGSOutput = vertexColorVSOutput[0];

    // emit four vertices needed to draw a billboard using triangle strips
    for (int j = 0; j < 4; j++) {
        gl_Position = vertices[j];
        EmitVertex ();
        }
    EndPrimitive();

}
