vec3 evaluateSheen(float sheen, float sheenTint, vec3 tint,
    float schlickWeight)
{
    vec3 tintMix = mix(vec3(1), tint, sheenTint);
    return sheen * tintMix * schlickWeight;
}
