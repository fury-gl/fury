vec3 linear2Gamma(vec3 color)
{
    return pow(color, vec3(1 / 2.2));
}
