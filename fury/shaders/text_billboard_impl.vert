// Warning: don't remove the any comment which
// starts with STR_REPLACEMENT. These lines
// are replaced by the shader compiler to 
// improve the performance of the shader

normalizedVertexMCVSOutput = vertexMC.xyz - center + vPadding; // 1st Norm. [-scale, scale]
float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
float size = abs(normalizedVertexMCVSOutput.x) * 2;
normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1]
centerVertexMCVSOutput = (center-vPadding);
vec2 billboardSize = vec2(size*vRelativeSize.x, size*vRelativeSize.y ); // Fixes the scaling issue
vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

vec3 vertexPositionMC = (center - vPadding) +
    cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x +
    cameraUpMC * billboardSize.y* normalizedVertexMCVSOutput.y;
gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);
UV = vUV;
//STR_REPLACEMENT::borderColor = vBorderColor;
