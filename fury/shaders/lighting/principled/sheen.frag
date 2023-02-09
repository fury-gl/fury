vec3 evaluateSheen(float sheen, float sheenTint, vec3 baseColor, float dotHL)
{
    vec3 tint = calculateTint(baseColor);
    float fw = schlickWeight(dotHL);
    vec3 tintMix = mix(vec3(1), tint, sheenTint);
    return sheen * tintMix * fw;
}