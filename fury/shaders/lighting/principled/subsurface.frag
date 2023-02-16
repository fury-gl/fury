float evaluateSubsurface(float roughness, float schlickWeightL,
    float schlickWeightV, float dotHL, float dotLN, float dotNV)
{
    float fss90 = square(dotHL) * roughness;
    float fss = mix(1, fss90, schlickWeightL) * mix(1, fss90, schlickWeightV);
    float ss = 1.25 * (fss * (1 / (dotLN + dotNV) - .5) + .5);
    return ss;
}
