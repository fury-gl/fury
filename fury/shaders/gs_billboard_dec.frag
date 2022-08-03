in float scale;
in vec3 center;
in vec3 point;
uniform mat4 MCDCMatrix;
uniform mat4 WCDCMatrix;
uniform mat4 MCVCMatrix;

// Ported from:
// https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c
float sphIntersect( vec3 ro, vec3 rd, vec4 sph )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w * sph.w;
    float h = b*b - c;
    if( h < 0.0 ) return -1.0;
    h = sqrt( h );
    return -b - h;
}