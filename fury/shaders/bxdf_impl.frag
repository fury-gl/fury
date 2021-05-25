/*
vec3 ior = vec3(2.4);
vec3 F0_t = abs((1. - ior) / (1. + ior));
F0_t *= F0_t;
F0_t = mix(F0_t, albedo, metallic);

vec3 Lo_t = vec3(.0);

vec3 F_t = F_Schlick(1., F0_t);
vec3 specular_t = D * Vis * F_t;
vec3 diffuse_t = (1. - metallic) * (1. - F_t) * DiffuseLambert(albedo);
Lo_t += (diffuse_t + specular_t) * lightColor0 * NdV;

vec3 kS_t = F_SchlickRoughness(max(NdV, .0), F0_t, roughness);
vec3 kD_t = 1. - kS_t;
kD_t *= 1. - metallic;
vec3 ambient_t = (kD_t * irradiance * albedo + prefilteredColor * (kS_t * brdf.r + brdf.g));
vec3 color_t = ambient_t + Lo_t;
color_t = mix(color_t, color_t * ao, aoStrengthUniform);
color_t += emissiveColor;
color_t = pow(color_t, vec3(1. / 2.2));
fragOutput0 = vec4(color_t, opacity);
*/
// Irradiance from texture
//fragOutput0 = vec4(irradiance, opacity);

// Texture reflection + roughness (1 = irradiance, 0 fully reflective)
//fragOutput0 = vec4(prefilteredColor, opacity);

// Color + irradiance + reflection
//fragOutput0 = vec4(ambient, opacity);

// Possible baseColor
//fragOutput0 = vec4(albedo, opacity);

// Disney's Principled BRDF
vec3 sheenV3 = evaluateSheen(sheen, sheenTint, albedo, NdV);
//fragOutput0 = vec4(albedo + sheenV3, opacity);
//fragOutput0 = vec4(ambient + sheenV3, opacity);
fragOutput0 = vec4(color + sheenV3, opacity);
