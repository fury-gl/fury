vec3 srgbToLinearRgb(vec3 srgb)
{
    return vec3(srgbToLinear(srgb.r), srgbToLinear(srgb.g), srgbToLinear(srgb.b));
}
