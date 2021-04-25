/* Calculating the distance between the fragment to the center
   of the billboard */
/* Billboard  Fragment shader implementation */
// Renaming variables passed from the Vertex Shader
vec3 color = vertexColorVSOutput.rgb;
vec3 point = normalizedVertexMCVSOutput;
fragOutput0 = vec4(color, 1.);

float len = length(point);
float radius = 1.;
float s = 0.5;

//float vMarker = 2.;
vec3 result = getDistFunc(point.xy, s, vEdgeWidth, vMarker);
float sdf = result.x;
float minSdf = result.y;
float edgeWidth = result.z;

if (sdf<0.0) discard;

vec3 color2 = vEdgeColor;
vec4 rgba = vec4(  color, 1 );
if (edgeWidth > 0.0){
   if (sdf < edgeWidth)  rgba  = vec4(color2, 0.5);
}
   

fragOutput0 = rgba;