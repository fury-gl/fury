//centerNew += vPadding;
// vec3 re = vec3(vRelativeSize.x, 1.0, vRelativeSize.y);
//vec3 re = vec3(1.0, 1.0, 1.0);

//centerVertexMCVSOutput = center;
normalizedVertexMCVSOutput = vertexMC.xyz - center + vPadding; // 1st Norm. [-scale, scale]
float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
float size = abs(normalizedVertexMCVSOutput.x) * 2;
normalizedVertexMCVSOutput *= scalingFactor; // 2nd Norm. [-1, 1]
centerVertexMCVSOutput = (center-vPadding);
// vec2 billboardSize = vec2(size, size); // Fixes the scaling issue
vec2 billboardSize = vec2(size*vRelativeSize.x, size*vRelativeSize.y ); // Fixes the scaling issue
vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);
// vec3 vertexPositionMC = center - vPadding +
//vec3 vertexPositionMC = center +
vec3 vertexPositionMC = (center - vPadding) +
    cameraRightMC * billboardSize.x * normalizedVertexMCVSOutput.x +
    cameraUpMC * billboardSize.y* normalizedVertexMCVSOutput.y;
gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);
//gl_Position = gl_Position*vec4(1.0, 1.0, 0, 1.0);
UV = vUV;
//borderColor = vBorderColor;