vec3 evaluateSpecularAnisotropic(float specular, float specularTint,
    float metallic, float anisotropic, float roughness, vec3 tint,
    vec3 color, float schlickWeight, float dotHN, float dotHX, float dotHY,
    float dotLN, float dotLX, float dotLY, float dotNV, float dotVX,
    float dotVY)
{
    vec3 tintMix = mix(vec3(1), tint, specularTint);
    vec3 spec = mix(specular * .08 * tintMix, color, metallic);

    float aspect = sqrt(1 - anisotropic * .9);

    float ax = max(.001, square(roughness) / aspect);
    float ay = max(.001, square(roughness) * aspect);

    float d = GTR2Anisotropic(ax, ay, dotHN, dotHX, dotHY);

    vec3 f = mix(spec, vec3(1), schlickWeight);

    float g = smithGGXAnisotropic(ax, ay, dotLN, dotLX, dotLY);
    g *= smithGGXAnisotropic(ax, ay, dotNV, dotVX, dotVY);

    return d * f * g;
}
