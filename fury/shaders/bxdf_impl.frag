// Possible baseColor
//fragOutput0 = vec4(albedo, opacity);

// Irradiance from texture
//fragOutput0 = vec4(irradiance, opacity);

// Light heatmap
//fragOutput0 = vec4(worldReflect, opacity);

// Texture reflection + roughness (1 = irradiance, 0 fully reflective)
//fragOutput0 = vec4(prefilteredColor, opacity);

//fragOutput0 = vec4(brdf, 0, opacity);

//fragOutput0 = vec4(specular, opacity);

// White + Normal (Black edge)
//fragOutput0 = vec4(lightColor0 * NdV, opacity);

// Reflection / Specular fraction
//vec3 kS = F_SchlickRoughness(max(NdV, 0.0), F0, roughness);
// Refraction / Diffuse fraction
//vec3 kD = 1.0 - kS;

// Color + irradiance + reflection
//vec3 ambientV3 = irradiance * albedo + prefilteredColor * (brdf.r + brdf.g);
//fragOutput0 = vec4(ambient, opacity);
//fragOutput0 = vec4(ambientV3, opacity);

// Disney's Principled BRDF
// Subsurface
vec3 LoV3 = vec3(.0);
// VTK's diffuse produces a similar effect
vec3 diffuseV3 = evaluateDiffuse(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(diffuseV3, opacity);

//fragOutput0 = vec4(diffuse, opacity);
diffuseV3 *= (1. - F);
//diffuseV3 *= (1. - metallic) * (1. - F);
//fragOutput0 = vec4(diffuseV3, opacity);

LoV3 += diffuseV3 * lightColor0 * NdV;
//LoV3 += (diffuseV3 + specular) * lightColor0 * NdV;
//fragOutput0 = vec4(Lo, opacity);
//fragOutput0 = vec4(LoV3, opacity);

// Isotropic BRDF
vec3 specularV3 = evaluateMicrofacetIsotropic(specularValue, specularTint,
        metallic, roughness, albedo, max(NdV, .0), NdV, NdV, NdV);
//fragOutput0 = vec4(specularV3, opacity);

//LoV3 += (diffuseV3 + specularV3) * lightColor0 * NdV;

// Anisotropic BRDF
/*
vec3 spec = evaluateBRDF(anisotropic, roughness, NdV, NdV, NdV, NdV, NdV, NdV,
        NdV, NdV, NdV);
*/
vec3 tangent = vec3(.0);
vec3 binormal = vec3(.0);
directionOfAnisotropicity(N, tangent, binormal);
//createBasis(N, tangent, binormal);
//fragOutput0 = vec4(N, opacity);
//fragOutput0 = vec4(tangent, opacity);
//fragOutput0 = vec4(binormal, opacity);

float HdX = clamp(dot(V, tangent), 1e-5, 1.);
float HdY = clamp(dot(V, binormal), 1e-5, 1.);
//fragOutput0 = vec4(vec3(HdX, .0, .0), opacity);
//fragOutput0 = vec4(vec3(.0, .0, HdY), opacity);

vec3 anisotropicV3 = evaluateMicrofacetAnisotropic(
        specularValue, specularTint, metallic, anisotropic, roughness, albedo,
        1., NdV, HdX, HdY, NdV, HdX, HdY, NdV, HdX, HdY);
fragOutput0 = vec4(anisotropicV3, opacity);

//LoV3 += (diffuseV3 + anisotropicV3) * lightColor0 * NdV;

// Subsurface
vec3 subsurfaceV3 = evaluateSubsurface(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(subsurfaceV3, opacity);

LoV3 = mix(LoV3, subsurfaceV3, subsurfaceV3);
//fragOutput0 = vec4(LoV3, opacity);

// Sheen + Sheen Tint
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, NdV);
//fragOutput0 = vec4(sheenV3, opacity);

LoV3 += sheenV3;

LoV3 *= (1. - metallic);

//LoV3 += anisotropicV3;
LoV3 += specularV3;

// Clearcoat + Clearcoat Gloss
float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, max(NdV, .0),
        NdV, NdV, NdV);
//fragOutput0 = vec4(vec3(clearcoatF), opacity);

LoV3 += clearcoatF;

vec3 colorV3 = ambient + LoV3;
//vec3 colorV3 = ambientV3 + LoV3;
colorV3 = mix(colorV3, colorV3 * ao, aoStrengthUniform);
colorV3 += emissiveColor;
// HDR tonemapping
//colorV3 = colorV3 / (colorV3 + vec3(1.));
// Gamma correction
colorV3 = pow(colorV3, vec3(1. / 2.2));
fragOutput0 = vec4(colorV3, opacity);
