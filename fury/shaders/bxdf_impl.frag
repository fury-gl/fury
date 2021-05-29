// Irradiance from texture
//fragOutput0 = vec4(irradiance, opacity);

// Texture reflection + roughness (1 = irradiance, 0 fully reflective)
//fragOutput0 = vec4(prefilteredColor, opacity);

// Color + irradiance + reflection
//fragOutput0 = vec4(ambient, opacity);

// Possible baseColor
//fragOutput0 = vec4(albedo, opacity);

// Disney's Principled BRDF
// Diffuse
vec3 diffuseV3 = evaluateDiffuse(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(diffuseV3, opacity);

// Sheen + Sheen Tint
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, NdV);
//fragOutput0 = vec4(albedo + sheenV3, opacity);
//fragOutput0 = vec4(ambient + sheenV3, opacity);
//fragOutput0 = vec4(color + sheenV3, opacity);

// Clearcoat + Clearcoat Gloss
float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, NdV, NdV, NdV, NdV);
fragOutput0 = vec4(color + sheenV3 + clearcoatF, opacity);

// Anisotropic
/*
vec3 spec = evaluateBRDF(anisotropic, roughness, NdV, NdV, NdV, NdV, NdV, NdV,
        NdV, NdV, NdV);
*/
vec3 anisotropicV3 = evaluateMicrofacetAnisotropic(
        specularValue, specularTint, metallic, anisotropic, roughness, albedo,
        NdV, NdV, NdV, NdV, NdV, NdV, NdV, NdV, NdV, NdV);
//fragOutput0 = vec4(ambient + sheenV3 + anisotropicV3 + clearcoatF, opacity);
