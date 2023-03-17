/* Billboard  vertex shader implementation */
centerVertexMCVSOutput = center;
normalizedVertexMCVSOutput = bbNorm(vertexMC.xyz, center);
float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
float size = abs((vertexMC.xyz - center).x) * 2;
vec2 shape = vec2(size, size); // Fixes the scaling issue
