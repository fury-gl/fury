// Ported from:
// https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c
float sphIntersect(vec3 ro, vec3 rd, vec3 sph, float r)
{
    vec3 oc = ro - sph;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - r * r;
    float h = b*b - c;
    if( h < 0.0 ) return -1.0;
    h = sqrt( h );
    return -b - h;
}