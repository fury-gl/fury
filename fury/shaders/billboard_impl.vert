/* Billboard  vertex shader implementation */
//VTK::ValuePass::Impl


centeredVertexMC = vertexMC.xyz - center;
scalingFactor = 1. / abs(centeredVertexMC.x);
centeredVertexMC *= scalingFactor;

vec2 billboard_size = vec2(scalingFactor, scalingFactor);

vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPosition_worldspace = center + CameraRight_worldspace * billboard_size.x * centeredVertexMC.x + CameraUp_worldspace * billboard_size.y * centeredVertexMC.y;
gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);
