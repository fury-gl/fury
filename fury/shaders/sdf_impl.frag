/* SDF fragment shader implementation */

//VKT::Light::Impl

vec3 point = vertexMCVSOutput.xyz;

// Ray Origin
// Camera position in world space
vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;

// Ray Direction
vec3 rd = normalize(pnt - ro);

// Light Direction
vec3 ld = normalize(ro - pnt);

ro += pnt - ro;

float t = castRay(ro, rd);

if(t < 20)
{
    vec3 pos = ro + t * rd;
    vec3 normal = centralDiffsNormals(pos, .0001);
    // Light Attenuation
    float la = dot(ld, normal);
    vec3 color = blinnPhongIllumModel(la, lightColor0,
        diffuseColor, specularPower, specularColor, ambientColor);
    fragOutput0 = vec4(color, opacity);
}
else
{
  discard;
}
