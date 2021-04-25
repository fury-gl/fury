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
float sdf = 0.0;
float minSdf = 0.5/2.0;
vec2 ds = abs(point.xy) - vec2(s, s);
sdf = -length(max(ds,0.0)) - min(max(ds.x,ds.y),0.0);

if (sdf<0.0) discard;

vec3 color2 = vEdgeColor;
vec4 rgba = vec4(  color, 1 );
if (vEdgeWidth > 0.0){
   if (sdf < vEdgeWidth)  rgba  = vec4(color2, 0.5);
}
   

fragOutput0 = rgba;