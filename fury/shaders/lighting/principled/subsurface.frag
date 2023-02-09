float evaluateSubsurface(float roughness, float dotLN, float dotNV,
    float dotHL)
{
    float fwl = schlickWeight(dotLN);
    float fwv = schlickWeight(dotNV);

    float fss90 = square(dotHL) * roughness;
    float fss = mix(1, fss90, fwl) * mix(1, fss90, fwv);
    float ss = 1.25 * (fss * (1 / (dotLN + dotNV) - .5) + .5);
    return ss;
}
