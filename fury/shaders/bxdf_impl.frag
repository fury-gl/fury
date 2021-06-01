// Irradiance from texture
//fragOutput0 = vec4(irradiance, opacity);

// Texture reflection + roughness (1 = irradiance, 0 fully reflective)
//fragOutput0 = vec4(prefilteredColor, opacity);

// Color + irradiance + reflection
//fragOutput0 = vec4(ambient, opacity);

// Possible baseColor
//fragOutput0 = vec4(albedo, opacity);

// White + Normal (Black edge)
//fragOutput0 = vec4(lightColor0 * NdV, opacity);

// Disney's Principled BRDF
// Subsurface
// VTK's diffuse produces a similar effect
vec3 diffuseV3 = evaluateDiffuse(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(diffuse, opacity);
//fragOutput0 = vec4(diffuseV3, opacity);
vec3 LoV3 = diffuseV3 * lightColor0 * NdV;
//fragOutput0 = vec4(Lo, opacity);
//fragOutput0 = vec4(LoV3, opacity);
//vec3 LoV3TMP = (diffuseV3 + specular) * lightColor0 * NdV;
//vec3 ambientV3 = irradiance * albedo + prefilteredColor * (brdf.r + brdf.g);
//fragOutput0 = vec4(ambient, opacity);
//fragOutput0 = vec4(ambientV3, opacity);
vec3 colorV3 = ambient + LoV3;
//vec3 colorV3 = ambientV3 + LoV3;
colorV3 = mix(colorV3, colorV3 * ao, aoStrengthUniform);
colorV3 += emissiveColor;
colorV3 = pow(colorV3, vec3(1. / 2.2));
//fragOutput0 = vec4(color, opacity);
//fragOutput0 = vec4(colorV3, opacity);
//fragOutput0 = vec4(colorV3 * (1. - metallic), opacity);
vec3 subsurfaceV3 = evaluateSubsurface(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(subsurfaceV3, opacity);
//fragOutput0 = vec4(mix(diffuseV3, subsurfaceV3, subsurface), opacity);
//fragOutput0 = vec4(mix(color, subsurfaceV3, subsurface), opacity);
//fragOutput0 = vec4(mix(colorV3, subsurfaceV3, subsurface) * (1. - metallic) + specular, opacity);

// Sheen + Sheen Tint
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, NdV);
//fragOutput0 = vec4(albedo + sheenV3, opacity);
//fragOutput0 = vec4(ambient + sheenV3, opacity);
//fragOutput0 = vec4(color + sheenV3, opacity);
//fragOutput0 = vec4((mix(color, subsurfaceV3, subsurface) + sheenV3), opacity);

// Clearcoat + Clearcoat Gloss
float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, NdV, NdV, NdV, NdV);
//fragOutput0 = vec4(color + sheenV3 + clearcoatF, opacity);
fragOutput0 = vec4((mix(color, subsurfaceV3, subsurface) + sheenV3) + clearcoatF, opacity);

// Anisotropic
/*
vec3 spec = evaluateBRDF(anisotropic, roughness, NdV, NdV, NdV, NdV, NdV, NdV,
        NdV, NdV, NdV);
*/
vec3 tangent = vec3(.0);
vec3 binormal = vec3(.0);
float NdX = clamp(dot(N, tangent), .0, 1.);
float NdY = clamp(dot(N, binormal), .0, 1.);
vec3 anisotropicV3 = evaluateMicrofacetAnisotropic(
        specularValue, specularTint, metallic, anisotropic, roughness, albedo,
        NdV, NdV, NdX, NdY, NdV, NdX, NdY, NdV, NdX, NdY);
//fragOutput0 = vec4(color + anisotropicV3, opacity);
