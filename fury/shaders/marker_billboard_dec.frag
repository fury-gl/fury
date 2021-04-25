/* We need both the ModelView and projection matrices */
in vec3 centerVertexMCVSOutput;
in vec3 normalizedVertexMCVSOutput;
in float vEdgeWidth;
in vec3 vEdgeColor;
uniform mat4 MCDCMatrix;
uniform mat4 MCVCMatrix;

float ndot(vec2 a, vec2 b ) {
    return a.x*b.x - a.y*b.y;
}