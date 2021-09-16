Lo = vec3(.0);
radiance = lightColor0;

// Diffuse
// VTK's diffuse produces a similar effect
diffuse = evaluateDiffuse(roughness, albedo, NdV, NdV, NdV);
//fragOutput0 = vec4(diffuse, opacity);

diffuse *= (1. - F);
//diffuse *= (1. - metallic) * (1. - F);
//fragOutput0 = vec4(diffuse, opacity);

Lo += diffuse;
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
//fragOutput0 = vec4(Lo, opacity);

Lo *= (1. - metallic);
//fragOutput0 = vec4(Lo, opacity);

// Isotropic BRDF
//specular = evaluateMicrofacetIsotropic(specularIntensity, specularTint,
//    metallic, roughness, albedo, max(NdV, .0), NdV, NdV, NdV);
//fragOutput0 = vec4(specular, opacity);

// Anisotropic BRDF
//specular = evaluateBRDF(anisotropic, roughness, NdV, NdV, NdV, NdV, NdV, NdV,
//    NdV, NdV, NdV);
//fragOutput0 = vec4(specular, opacity);
vec3 tangent = vec3(.0);
vec3 binormal = vec3(.0);
directionOfAnisotropicity(N, tangent, binormal);
//createBasis(N, tangent, binormal);
//fragOutput0 = vec4(tangent, opacity);
//fragOutput0 = vec4(binormal, opacity);

float VdX = clamp(dot(V, tangent), 1e-5, 1.);
float VdY = clamp(dot(V, binormal), 1e-5, 1.);

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

VdX = clamp(dot(V, mix(N, anisotropicTangent, anisotropic)), 1e-5, 1.);
VdY = clamp(dot(V, mix(N, anisotropicNormal, anisotropic)), 1e-5, 1.);

//F0 = mix(vec3(baseF0Uniform), albedo, metallic);
// specular occlusion, it affects only material with an f0 < 0.02, else f90 is 1.0
float f90 = clamp(dot(F0, vec3(50.0 * 0.33)), 0.0, 1.0);
//vec3 F90 = mix(vec3(f90), edgeTintUniform, metallic);
vec3 F90 = mix(vec3(f90), albedo, metallic);
//fragOutput0 = vec4(F90, opacity);

//specular = vtkSpecularAnisotropic(anisotropic, roughness, 1., NdV, HdX, HdY,
//        NdV, HdX, HdY, NdV, HdX, HdY, F0, F90);
//fragOutput0 = vec4(specular, opacity);

specular = evaluateMicrofacetAnisotropic(specularIntensity, specularTint,
    metallic, anisotropic, roughness, albedo, 1., NdV, VdX, VdY, NdV, VdX, VdY,
    NdV, VdX, VdY);
//fragOutput0 = vec4(specular, opacity);

Lo += specular;
//fragOutput0 = vec4(Lo, opacity);

// Clearcoat + Clearcoat Gloss
// TODO: Add Clearcoat Normal
float coatNdV = clamp(dot(N, V), 1e-5, 1.0);
vec3 coatWorldReflect = normalize(envMatrix * reflect(-V, N));

// TODO: Check if Gloss must be inverted
vec3 prefilteredSpecularCoatColor = textureLod(prefilterTex, coatWorldReflect, clearcoatGloss * prefilterMaxLevel).rgb;
vec2 coatBrdf = texture(brdfTex, vec2(coatNdV, clearcoatGloss)).rg;
//fragOutput0 = vec4(coatWorldReflect, opacity);
//fragOutput0 = vec4(prefilteredSpecularCoatColor, opacity);

// TODO: Clearcoat F0 and F90

// TODO: Check if SpecularIsotropic does the same
float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, max(NdV, .0),
        NdV, NdV, NdV);
//fragOutput0 = vec4(vec3(clearcoatF), opacity);

// TODO: Energy compensation

Lo += clearcoatF;

Lo *= radiance * NdV;

vec3 specularBrdf = F0 * brdf.r + F90 * brdf.g;
//fragOutput0 = vec4(specularBrdf, opacity);
vec3 iblSpecular = prefilteredSpecularColor * specularBrdf;
//fragOutput0 = vec4(iblSpecular, opacity);

vec3 iblDiffuse = (1.0 - F0) * (1.0 - metallic) * irradiance * albedo;
//fragOutput0 = vec4(iblDiffuse, opacity);

// TODO: Clearcoat attenuation

color = iblDiffuse + iblSpecular;
//fragOutput0 = vec4(color, opacity);

color += Lo;
color = mix(color, color * ao, aoStrengthUniform);
color += emissiveColor;
// Gamma correction
color = pow(color, vec3(1. / 2.2));
fragOutput0 = vec4(color, opacity);
