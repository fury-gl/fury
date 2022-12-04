in vec3 center;
in vec3 diff;

out vec3 centerVertexMCVSOutput;

vec3 orient2rgb(vec3 v)
{
    float r = sqrt(dot(v, v));
    if (r != 0)
        return abs(v / r);
    return vec3(1);
}
