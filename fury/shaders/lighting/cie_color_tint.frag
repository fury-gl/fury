vec3 calculateTint(vec3 color)
{
    float lum = dot(vec3(.3, .6, .1), color);  // Relative Luminance approx.
    return lum > 0 ? color / lum : vec3(1);  // Lum. norm. to isolate hue + sat
}
