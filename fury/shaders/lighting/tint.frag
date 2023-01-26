vec3 tint(vec3 albedo)
{
    float lum = dot(vec3(.3, .6, .1), albedo);
    return lum > .0 ? albedo / lum : vec3(1);
}
