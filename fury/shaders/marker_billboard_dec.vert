/* Billboard  vertex shader declaration */
in vec3 center;

out vec3 centerVertexMCVSOutput;
out vec3 normalizedVertexMCVSOutput;

in float edgeWidth;
in vec3 edgeColor;

out vec3 vEdgeColor;
out float vEdgeWidth;