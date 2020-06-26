/* SDF vertex shader declaration */

//VTK::ValuePass::Dec

in vec3 center;
in int primitive;

out vec4 vertexMCVSOutput;

out vec3 centerWCVSOutput;
flat out int primitiveVSOutput;