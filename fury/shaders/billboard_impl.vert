/* Billboard  vertex shader implementation */
centeredVertexMC = vertexMC.xyz - center; // 1st Norm. [-scale, scale]
scalingFactor = 1. / abs(centeredVertexMC.x);
float size = abs(centeredVertexMC.x) * 2;
centeredVertexMC *= scalingFactor; // 2nd Norm. [-1, 1]
vec2 billboardSize = vec2(size, size); // Fixes the scaling issue
vec3 cameraRightWorldSpace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpWorldSpace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPositionWorldSpace = center +
cameraRightWorldSpace * billboardSize.x * centeredVertexMC.x +
cameraUpWorldSpace * billboardSize.y * centeredVertexMC.y;
gl_Position = MCDCMatrix * vec4(vertexPositionWorldSpace, 1.);
