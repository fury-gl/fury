vec3 evaluateSpecularIsotropic(float specular, float specularTint,
    float metallic, float roughness, vec3 tint, vec3 color,
    float schlickWeight, float dotHN, float dotLN, float dotNV)
{
    vec3 tintMix = mix(vec3(1), tint, specularTint);
    vec3 spec = mix(specular * .08 * tintMix, color, metallic);

    float a = max(.001, square(roughness));

    float d = GTR2(dotHN, a);

    vec3 f = mix(spec, vec3(1), schlickWeight);

    float g = smithGGX(dotLN, a) * smithGGX(dotNV, a);

    return d * f * g;
}
