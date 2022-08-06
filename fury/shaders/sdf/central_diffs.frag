vec3 centralDiffsNormals(in vec3 p, float eps)
{
    vec2 h = vec2(eps, 0);
    return normalize(vec3(map(p + h.xyy) - map(p - h.xyy),
                          map(p + h.yxy) - map(p - h.yxy),
                          map(p + h.yyx) - map(p - h.yyx)));
}
