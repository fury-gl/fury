/*
 * The goal of this shader is to increase simulate billboard
 */

//VTK::ValuePass::Dec
in vec3 center;
uniform mat4 Ext_mat;
out vec3 centeredVertexMC;
out vec3 cameraPosition;
out vec3 viewUp;