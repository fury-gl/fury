float evaluateClearcoat(float clearcoat, float clearcoatGloss,
    float schlickWeight, float dotHN, float dotLN, float dotNV)
{
    float gloss = mix(.1, .001, clearcoatGloss);
    float d = GTR1(gloss, dotHN);
    float f = mix(.04, 1, schlickWeight);
    float g = smithGGX(.25, dotLN) * smithGGX(.25, dotNV);
    return .25 * clearcoat * d * f * g;
}
