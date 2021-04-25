/* Billboard  vertex shader declaration */
in vec3 center;

out vec3 centerVertexMCVSOutput;
out vec3 normalizedVertexMCVSOutput;

//in float edgeWidth;
uniform float edgeWidth;
uniform vec3 edgeColor;
in float marker;

out vec3 vEdgeColor;
out float vEdgeWidth;
out float vMarker;