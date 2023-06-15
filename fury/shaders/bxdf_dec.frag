// Disney's Principled BXDF
#define EPSILON .0001
#define PI 3.14159265359

uniform float subsurface;
uniform float metallic;
uniform float specularTint;
uniform float roughness;
uniform float anisotropic;
uniform float sheen;
uniform float sheenTint;
uniform float clearcoat;
uniform float clearcoatGloss;

uniform float ambientOcclusion;

uniform vec3 subsurfaceColor;
uniform vec3 anisotropicDirection;

vec3 calculateFresnelSchlick(float dotHV, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - dotHV, 5.0);
}

float schlickWeight(float cosTheta)
{
    float m = clamp(1. - cosTheta, .0, 1.);
    return m * m * m * m * m;
}

vec3 calculateTint(vec3 baseColor)
{
    float luminance = dot(vec3(.3, .6, .1), baseColor);
    return luminance > .0 ? baseColor / luminance : vec3(1.);
}

void directionOfAnisotropicity(vec3 normal, out vec3 tangent, out vec3 binormal)
{
    tangent = cross(normal, anisotropicDirection);
    binormal = normalize(cross(normal, tangent));
    tangent = normalize(cross(normal, binormal));
}

float square(float x)
{
    return x * x;
}

float GTR1(float dotHN, float alpha)
{
    if(alpha >= 1.)
        return 1. / PI;
    float alpha2 = square(alpha);
    float dotHN2 = square(dotHN);
    float t = 1. + (alpha2 - 1.) * dotHN2;
    return (alpha2 - 1.) / (PI * log(alpha2) * t);
}

float GTR2(float dotHN, float alpha)
{
    float alpha2 = square(alpha);
    float dotHN2 = square(dotHN);
    float t = 1. + (alpha2 - 1.) * dotHN2;
    return alpha2 / (PI * square(t));
}

float GTR2Anisotropic(float dotHN, float dotHX, float dotHY, float ax,
                      float ay)
{
    float dotHN2 = square(dotHN);
    float dotHX2 = square(dotHX);
    float dotHY2 = square(dotHY);
    float ax2 = square(ax);
    float ay2 = square(ay);
    return 1. / (PI * ax * ay * square(dotHX2 / ax2 + dotHY2 / ay2 + dotHN2));
}

float smithGGGX(float dotNV, float alpha)
{
    float alpha2 = square(alpha);
    float b = square(dotNV);
    return 1. / (abs(dotNV) + max(sqrt(alpha2 + b - alpha2 * b), EPSILON));
}

float smithGGGXAnisotropic(float dotNV, float dotVX, float dotVY, float ax,
                           float ay)
{
    float dotVX2 = square(dotVX);
    float dotVY2 = square(dotVY);
    float ax2 = square(ax);
    float ay2 = square(ay);
    return 1. / (dotNV + sqrt(dotVX2 * ax2 + dotVY2 * ay2 + square(dotNV)));
}

float evaluateClearcoat(float clearcoat, float clearcoatGloss, float dotHL,
                        float dotHN, float dotLN, float dotNV)
{
    if(clearcoat <= .0)
        return .0;

    float gloss = mix(.1, .001, clearcoatGloss);
    float dr = GTR1(abs(dotHN), gloss);
    float fh = schlickWeight(dotHL);
    float fr = mix(.04, 1., fh);
    float gr = smithGGGX(dotLN, .25) * smithGGGX(dotNV, .25);
    return 1. * clearcoat * fr * gr * dr;
}

vec3 evaluateDiffuse(float roughness, vec3 baseColor, float dotHL,
                     float dotLN, float dotNV)
{
    float fl = schlickWeight(dotLN);
    float fv = schlickWeight(dotNV);

    float fd90 = .5 + 2. * square(dotHL) * roughness;
    float fd = mix(1., fd90, fl) * mix(1., fd90, fv);
    return (1. / PI) * fd * baseColor;
}

vec3 evaluateMicrofacetAnisotropic(float specular, float specularTint,
                                   float metallic, float anisotropic,
                                   float roughness, vec3 baseColor,
                                   float dotHL, float dotHN, float dotHX,
                                   float dotHY, float dotLN, float dotLX,
                                   float dotLY, float dotNV, float dotVX,
                                   float dotVY)
{
    if(dotLN <= .0 || dotNV <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    vec3 tintMix = mix(vec3(1.), tint, specularTint);
    vec3 spec = mix(specular * .08 * tintMix, baseColor, metallic);

    float aspect = sqrt(1. - anisotropic * .9);

    float ax = max(.001, square(roughness) / aspect);
    float ay = max(.001, square(roughness) * aspect);

    float ds = GTR2Anisotropic(dotHN, dotHX, dotHY, ax, ay);

    float fh = schlickWeight(dotHL);
    vec3 fs = mix(spec, vec3(1.), fh);

    float gs = smithGGGXAnisotropic(dotLN, dotLX, dotLY, ax, ay);
    gs *= smithGGGXAnisotropic(dotNV, dotVX, dotVY, ax, ay);

    return gs * fs * ds;
}

vec3 evaluateMicrofacetIsotropic(float specular, float specularTint,
                                 float metallic, float roughness,
                                 vec3 baseColor, float dotHL, float dotHN,
                                 float dotLN, float dotNV)
{
    if(specular <= 0)
        return vec3(.0);
    if(dotLN <= .0 || dotNV <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    vec3 tintMix = mix(vec3(1.), tint, specularTint);
    vec3 spec = mix(specular * .08 * tintMix, baseColor, metallic);

    float a = max(.001, square(roughness));

    float ds = GTR2(dotHN, a);
    float fh = schlickWeight(dotHL);
    vec3 fs = mix(spec, vec3(1.), fh);
    float gs = smithGGGX(dotLN, a);
    gs *= smithGGGX(dotNV, a);
    return gs * fs * ds;
}

vec3 evaluateSheen(float sheen, float sheenTint, vec3 baseColor, float dotHL)
{
    if(sheen <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    float fh = schlickWeight(dotHL);
    vec3 tintMix = mix(vec3(1.), tint, sheenTint);
    return sheen * tintMix * fh;
}

vec3 evaluateSubsurface(float roughness, vec3 color, float dotLN,
                        float dotNV, float dotHL)
{
    float fl = schlickWeight(dotLN);
    float fv = schlickWeight(dotNV);

    float fss90 = square(dotHL) * roughness;
    float fss = mix(1., fss90, fl) * mix(1., fss90, fv);
    float ss = 1.25 * (fss * (1. / (dotLN + dotNV) - .5) + .5);
    return (1 / PI) * ss * color;
}
