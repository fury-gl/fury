void updateTanBitan(vec3 norm, vec3 dir, out vec3 tang, out vec3 bitang)
{
    tang = cross(norm, dir);
    bitang = normalize(cross(norm, tang));
    tang = normalize(cross(norm, bitang));
}
