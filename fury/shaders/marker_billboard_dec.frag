/* We need both the ModelView and projection matrices */
in vec3 centerVertexMCVSOutput;
in vec3 normalizedVertexMCVSOutput;
in float vEdgeWidth;
in vec3 vEdgeColor;
uniform mat4 MCDCMatrix;
uniform mat4 MCVCMatrix;

float ndot(vec2 a, vec2 b ) {
    return a.x*b.x - a.y*b.y;
};
vec3 getDistFunc(vec2 p, float s, float edgeWidth){
    edgeWidth = edgeWidth/2.;
    float minSdf = 0.5;
    float sdf = -length(p) + s;
    vec3 result = vec3(sdf, minSdf, edgeWidth);
    return result ;
};
