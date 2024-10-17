vec3 tonemap(vec3 linear)
{
    float maxChannel = max(max(1.0, linear.r), max(linear.g, linear.b));
    return linear * ((1.0 - 0.02 * log2(maxChannel)) / maxChannel);
}
