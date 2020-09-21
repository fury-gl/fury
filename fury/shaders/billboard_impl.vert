/* Billboard  vertex shader implementation */
normalizedVertexMCVSOutput = vertexMC.xyz - center; // 1st Norm. [-scale, scale]
float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
float size = abs(normalizedVertexMCVSOutput.x) * 2;
normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1]
vec2 billboardSize = vec2(size, size); // Fixes the scaling issue
vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
vec3 vertexPositionMC = center +
    cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x +
    cameraUpMC * billboardSize.y * normalizedVertexMCVSOutput.y;
gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);
