float evaluateDiffuse(float roughness, vec3 baseColor, float schlickWeightL,
    float schlickWeightV, float dotHL)
{
    float fd90 = .5 + 2 * square(dotHL) * roughness;
    float fd = mix(1, fd90, schlickWeightL) * mix(1, fd90, schlickWeightV);
    return fd;
}
