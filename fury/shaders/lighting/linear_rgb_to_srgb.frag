vec3 linearRgbToSrgb(vec3 linear)
{
    return vec3(linearToSrgb(linear.r), linearToSrgb(linear.g), linearToSrgb(linear.b));
}
