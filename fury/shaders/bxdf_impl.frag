// Disney's Principled BXDF
const float prefilterMaxLevel = float(4);

vec3 albedo = pow(diffuseColor, vec3(2.2));

float ao = 1.0;

vec3 emissiveColor = vec3(0.0);

vec3 N = normalVCVSOutput;

vec3 V = normalize(-vertexVC.xyz);

float dotNV = clamp(dot(N, V), 1e-5, 1.0);

vec3 irradiance = vec3(0.0);

vec2 brdf = vec2(0.0, 0.0);

vec3 Lo = vec3(.0);

vec3 radiance = lightColor0;

vec3 F0 = mix(vec3(0.04), albedo, metallic);

vec3 F = calculateFresnelSchlick(1.0, F0);

// Diffuse
diffuse = evaluateDiffuse(roughness, albedo, dotNV, dotNV, dotNV);

//diffuse *= (1. - F);
diffuse *= (1. - metallic) * (1. - F);

Lo += diffuse;

// Subsurface
vec3 subsurfaceV3 = evaluateSubsurface(roughness, subsurfaceColor, dotNV,
    dotNV, dotNV);

Lo = mix(Lo, subsurfaceV3, subsurface);

// Sheen + Sheen Tint
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, dotNV);

Lo += sheenV3;

//Lo *= (1. - metallic);

// Isotropic specular
//specular = evaluateMicrofacetIsotropic(specularIntensity, specularTint,
//    metallic, roughness, albedo, max(dotNV, .0), dotNV, dotNV, dotNV);

// Anisotropic specular
vec3 tangent = vec3(.0);
vec3 binormal = vec3(.0);
directionOfAnisotropicity(N, tangent, binormal);

float dotTV = dot(tangent, V);
float dotBV = dot(binormal, V);

vec3 prefilteredSpecularColor = vec3(0.0);

// Specular occlusion, it affects only material with an f0 < 0.02, else f90 is
//1.0
float f90 = clamp(dot(F0, vec3(50.0 * 0.33)), .0, 1.);

vec3 F90 = mix(vec3(f90), albedo, metallic);

specular = evaluateMicrofacetAnisotropic(specularIntensity, specularTint,
    metallic, anisotropic, roughness, albedo, 1., dotNV, dotTV, dotBV, dotNV,
    dotTV, dotBV, dotNV, dotTV, dotBV);

Lo += specular;

// Clearcoat + Clearcoat Gloss
float coatDotNV = clamp(dot(N, V), 1e-5, 1.0);

float clearcoatF = evaluateClearcoat(clearcoat, clearcoatGloss, max(dotNV, .0),
        dotNV, dotNV, dotNV);

Lo += clearcoatF;

Lo *= radiance * dotNV;

vec3 specularBrdf = F0 * brdf.r + F90 * brdf.g;

vec3 iblSpecular = prefilteredSpecularColor * specularBrdf;

// No diffuse for metals
vec3 iblDiffuse = (1.0 - F0) * (1.0 - metallic) * irradiance * albedo;

vec3 color = iblDiffuse + iblSpecular;

color += Lo;

color = mix(color, color * ao, ambientOcclusion);

color += emissiveColor;

// Gamma correction
color = pow(color, vec3(1. / 2.2));
fragOutput0 = vec4(color, opacity);
