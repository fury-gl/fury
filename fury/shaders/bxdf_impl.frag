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

//fragOutput0 = vec4(kS, opacity);
//fragOutput0 = vec4(kD, opacity);

//fragOutput0 = vec4(kS * brdf.r + brdf.g, opacity);

//fragOutput0 = vec4(Lo, opacity);

// Reflection / Specular fraction
//vec3 kS = F_SchlickRoughness(max(NdV, 0.0), F0, roughness);
// Refraction / Diffuse fraction
//vec3 kD = 1.0 - kS;

// Disney's Principled BRDF
// Diffuse
Lo = vec3(.0);
// VTK's diffuse produces a similar effect
vec3 diffuseV3 = evaluateDiffuse(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(diffuseV3, opacity);

//fragOutput0 = vec4(diffuse, opacity);
diffuseV3 *= (1. - F);
//diffuseV3 *= (1. - metallic) * (1. - F);
//fragOutput0 = vec4(diffuseV3, opacity);

Lo += diffuseV3;
//Lo += diffuseV3 * lightColor0 * NdV;
//Lo += (diffuseV3 + specular) * lightColor0 * NdV;
//fragOutput0 = vec4(Lo, opacity);

// Subsurface
vec3 subsurfaceV3 = evaluateSubsurface(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(subsurfaceV3, opacity);

Lo = mix(Lo, subsurfaceV3, subsurface);
//fragOutput0 = vec4(Lo, opacity);

// Sheen + Sheen Tint
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, NdV);
//fragOutput0 = vec4(sheenV3, opacity);

Lo += sheenV3;

Lo *= (1. - metallic);

// Isotropic BRDF
vec3 specularV3 = evaluateMicrofacetIsotropic(specularValue, specularTint,
        metallic, roughness, albedo, max(NdV, .0), NdV, NdV, NdV);
//fragOutput0 = vec4(specularV3, opacity);

//Lo += (diffuseV3 + specularV3) * lightColor0 * NdV;
//Lo += specularV3;

// Anisotropic BRDF
/*
vec3 spec = evaluateBRDF(anisotropic, roughness, NdV, NdV, NdV, NdV, NdV, NdV,
        NdV, NdV, NdV);
*/
vec3 tangent = vec3(.0);
vec3 binormal = vec3(.0);
directionOfAnisotropicity(N, tangent, binormal);
//createBasis(N, tangent, binormal);
//fragOutput0 = vec4(tangent, opacity);
//fragOutput0 = vec4(binormal, opacity);

float HdX = clamp(dot(V, tangent), 1e-5, 1.);
float HdY = clamp(dot(V, binormal), 1e-5, 1.);

vec3 anisotropicTangent = cross(binormal, V);
vec3 anisotropicNormal = cross(anisotropicTangent, binormal);
vec3 bentNormal = normalize(mix(N, anisotropicNormal, anisotropic));
worldReflect = normalize(envMatrix * reflect(-V, bentNormal));
vec3 prefilteredSpecularColor = textureLod(prefilterTex, worldReflect, roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(anisotropicTangent, opacity);
//fragOutput0 = vec4(anisotropicNormal, opacity);
//fragOutput0 = vec4(bentNormal, opacity);
//fragOutput0 = vec4(worldReflect, opacity);
//fragOutput0 = vec4(prefilteredSpecularColor, opacity);

vec3 anisotropicV3 = evaluateMicrofacetAnisotropic(
        specularValue, specularTint, metallic, anisotropic, roughness, albedo,
        1., NdV, HdX, HdY, NdV, HdX, HdY, NdV, HdX, HdY);
//fragOutput0 = vec4(anisotropicV3, opacity);

//Lo += (diffuseV3 + anisotropicV3) * lightColor0 * NdV;
//Lo += (diffuseV3 + anisotropicV3);
Lo += anisotropicV3;

// Clearcoat + Clearcoat Gloss
// TODO: Add Clearcoat Normal
float coatNdV = clamp(dot(N, V), 1e-5, 1.0);
vec3 coatWorldReflect = normalize(envMatrix * reflect(-V, N));

// TODO: Check if Gloss must be inverted
vec3 prefilteredSpecularCoatColor = textureLod(prefilterTex, coatWorldReflect, clearcoatGloss * prefilterMaxLevel).rgb;
vec2 coatBrdf = texture(brdfTex, vec2(coatNdV, clearcoatGloss)).rg;
//fragOutput0 = vec4(coatWorldReflect, opacity);
//fragOutput0 = vec4(prefilteredSpecularCoatColor, opacity);

float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, max(NdV, .0),
        NdV, NdV, NdV);
//fragOutput0 = vec4(vec3(clearcoatF), opacity);

Lo += clearcoatF;

Lo *= lightColor0 * NdV;

//F0 = mix(vec3(baseF0Uniform), albedo, metallic);
// specular occlusion, it affects only material with an f0 < 0.02, else f90 is 1.0
float f90 = clamp(dot(F0, vec3(50.0 * 0.33)), 0.0, 1.0);
//vec3 F90 = mix(vec3(f90), edgeTintUniform, metallic);
vec3 F90 = mix(vec3(f90), albedo, metallic);
//fragOutput0 = vec4(F90, opacity);

vec3 specularBrdf = F0 * brdf.r + F90 * brdf.g;
//fragOutput0 = vec4(specularBrdf, opacity);
vec3 iblSpecular = prefilteredSpecularColor * specularBrdf;
//fragOutput0 = vec4(iblSpecular, opacity);

vec3 iblDiffuse = (1.0 - F0) * (1.0 - metallic) * irradiance * albedo;
//fragOutput0 = vec4(iblDiffuse, opacity);

color = iblDiffuse + iblSpecular;
//fragOutput0 = vec4(color, opacity);

color += Lo;
color = mix(color, color * ao, aoStrengthUniform);
color += emissiveColor;
// Gamma correction
color = pow(color, vec3(1. / 2.2));
fragOutput0 = vec4(color, opacity);
