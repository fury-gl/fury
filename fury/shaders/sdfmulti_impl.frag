/* SDF fragment shader implementation */

//VKT::Light::Impl

vec3 point = vertexMCVSOutput.xyz;

//ray origin
vec4 ro = -MCVCMatrix[3] * MCVCMatrix;  // camera position in world space

//ray direction
vec3 rd = normalize(point - ro.xyz);

ro += vec4((point - ro.xyz),0.0);

//float t = castRay(ro.xyz, rd);
 vec2 res =  castRay(ro.xyz, rd);
 float t = res.x;
 float m = res.y;

if(t < 20.0)
{
    vec3 position = ro.xyz + t * rd;
    vec3 norm = calcNormal(position);

    fragOutput0 = vec4( norm, 1.0);

}
else{
   discard;
}
