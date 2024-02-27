vec3 srgb_to_linear_rgb(vec3 srgb)
{
    return vec3(srgb_to_linear(srgb.r), srgb_to_linear(srgb.g), srgb_to_linear(srgb.b));
}
