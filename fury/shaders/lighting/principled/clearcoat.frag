float evaluateClearcoat(float clearcoat, float clearcoatGloss,
    float schlickWeight, float dotHN, float dotLN, float dotNV)
{
    float gloss = mix(.1, .001, clearcoatGloss);
    float d = GTR1(dotHN, gloss);
    float f = mix(.04, 1, schlickWeight);
    float g = smithGGX(dotLN, .25) * smithGGX(dotNV, .25);
    return .25 * clearcoat * d * f * g;
}
