/*
 * The goal of this shader is to increase simulate billboard
 */


//VTK::ValuePass::Impl
centeredVertexMC = vertexMC.xyz - center;
float scalingFactor = 1. / abs(centeredVertexMC.x);
centeredVertexMC *= scalingFactor;
vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPosition_worldspace = center + CameraRight_worldspace * 0.5 * vertexMC.x + CameraUp_worldspace * -0.5 * vertexMC.y;
gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);