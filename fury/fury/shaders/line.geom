/*
 * The goal of this shader is to increase the realism of line
 * rendering by introducing the size depth cue. Unlike OpenGL
 * lines, the products of this shader will decrease in size
 * with distance from the camera. This effect is accomplished by
 * replacing each line segment with a camera-aligned rectangle.
 *
 * Borrows heavily from VTK's vtkPolyDataWideLineGS.glsl
 */

//VTK::System::Dec
//VTK::PositionVC::Dec
uniform float linewidth;
uniform mat4 MCDCMatrix;

//VTK::PrimID::Dec

// declarations below aren't necessary because
// they are already injected by PrimID template
// this comment is just to justify the passthrough below
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

// convert lines to triangle strips
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

void main() {
    // pass through vertex color to FS
    vertexColorGSOutput = vertexColorVSOutput[0];


    /*
     * Compute vectors for the depth cue
     */

    // compute line segment direction
    vec4 direction = normalize(gl_in[1].gl_Position - gl_in[0].gl_Position);

    // compute orthogonal vector in XY plane by rotating line 90 degrees
    vec4 ortho = vec4(-1.0 * direction.y, direction.x, direction.z, direction.w);

    // transform normal to MC
    vec4 orthoMC = inverse(MCDCMatrix) * ortho;


    /*
     * Compute vector for the fake tubes
     * Used by the VTK shader for wide lines
     */

    // compute line segment direction
    vec2 normal = normalize(
        gl_in[1].gl_Position.xy/gl_in[1].gl_Position.w -
        gl_in[0].gl_Position.xy/gl_in[0].gl_Position.w);

    // rotate 90 degrees
    normal = vec2(-1.0*normal.y,normal.x);

    //VTK::Normal::Start

    // create 4 new vertices in triangle strip
    // for each original vertex, emit 2 new vertices
    // and translate them in the normal direction
    for (int j = 0; j < 4; j++)
    {
        int i = j/2;

        //VTK::PrimID::Impl

        //VTK::Clip::Impl

        //VTK::Color::Impl

        //VTK::Normal::Impl

        //VTK::Light::Impl

        //VTK::TCoord::Impl

        //VTK::DepthPeeling::Impl

        //VTK::Picking::Impl

        // VC position of this fragment
        //VTK::PositionVC::Impl

        gl_Position = gl_in[i].gl_Position + 
                      (MCDCMatrix * linewidth * orthoMC * ((j + 1) % 2 - 0.5));

        EmitVertex();
    }

    EndPrimitive();
}
