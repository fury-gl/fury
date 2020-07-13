/* SDF vertex shader declaration */

//VTK::ValuePass::Dec

in vec3 center;
in float primitive;
in float scale;

out vec4 vertexMCVSOutput;

out vec3 centerWCVSOutput;
flat out int primitiveVSOutput;
out float scaleVSOutput;