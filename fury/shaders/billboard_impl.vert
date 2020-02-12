/* Billboard  vertex shader implementation */
//VTK::ValuePass::Impl


centeredVertexMC = vertexMC.xyz - center;
scalingFactor = 1. / abs(centeredVertexMC.x);
centeredVertexMC *= scalingFactor;

vec2 billboardSize = vec2(scalingFactor, scalingFactor);

vec3 cameraRightWorldSpace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpWorldSpace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPositionWorldSpace = center + cameraRightWorldSpace * billboardSize.x * centeredVertexMC.x + cameraUpWorldSpace * billboardSize.y * centeredVertexMC.y;
gl_Position = MCDCMatrix * vec4(vertexPositionWorldSpace, 1.);
