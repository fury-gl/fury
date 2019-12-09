/*
 * The goal of this shader is to increase simulate billboard
 */

// Renaming variables passed from the Vertex Shader
vec3 color = vertexColorVSOutput.rgb;
vec3 point = centeredVertexMC;
fragOutput0 = vec4(color, 0.7);


/*float len = length(point);
// VTK Fake Spheres
float radius = 1.;
if(len > radius)
    discard;
vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
vec3 direction = normalize(vec3(1., 1., 1.));
float df = max(0, dot(direction, normalizedPoint));
float sf = pow(df, 24);
fragOutput0 = vec4(max(df * color, sf * vec3(1)), 1);*/