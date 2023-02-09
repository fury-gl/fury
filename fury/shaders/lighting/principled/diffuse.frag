float evaluateDiffuse(float roughness, vec3 baseColor, float dotHL,
    float dotLN, float dotNV)
{
    float fwl = schlickWeight(dotLN);
    float fwv = schlickWeight(dotNV);

    float fd90 = .5 + 2 * square(dotHL) * roughness;
    float fd = mix(1, fd90, fwl) * mix(1, fd90, fwv);
    return fd;
}
