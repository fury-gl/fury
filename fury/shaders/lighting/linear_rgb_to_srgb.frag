vec3 linear_rgb_to_srgb(vec3 linear)
{
    return vec3(linear_to_srgb(linear.r), linear_to_srgb(linear.g), linear_to_srgb(linear.b));
}
