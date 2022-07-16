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
uniform mat4 MCVCMatrix;
uniform mat4 MCWCMatrix;
uniform mat4 VCMCMatrix;
uniform float scale;

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
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

void main() {



            const vec2 coordinates [] = vec2 [] (vec2 (-0.5f, -0.5f),
                                           vec2 (0.5f, -0.5f),
                                           vec2 (-0.5f, 0.5f),
                                           vec2 (0.5f, 0.5f)
                                           );

      float max_near_far_diff = 50;
      for (int j = 0; j < 4; j++) {
            gl_Position = MCDCMatrix * ( (gl_in [0].gl_Position ) +  vec4(coordinates[j], 0, 0));
            vertexColorGSOutput = vertexColorVSOutput[0];
            EmitVertex ();}

        EndPrimitive();




}
