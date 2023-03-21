vec3 bbNorm(vec3 vertex, vec3 center)
{
    vec3 normalizedVertex = vertex - center; // 1st Norm. [-scale, scale]
    float scalingFactor = 1. / abs(normalizedVertex.x);
    normalizedVertex *= scalingFactor; // 2nd Norm. [-1, 1]
    return normalizedVertex;
}
