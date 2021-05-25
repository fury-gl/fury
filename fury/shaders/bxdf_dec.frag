float chiGGX(float v)
{
    return v > 0 ? 1. : .0;
}

vec3 FresnelSchlick(float HdV, vec3 F0)
{
    return F0 + (1 - F0) * pow(1 - HdV, 5);
}

float GGXDistribution(float NdH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdH2 = NdH * NdH;
    float den = NdH2 * alpha2 + (1 - NdH2);
    return (chiGGX(NdH) * alpha2) / (PI * den * den);
}

float GGXPartialGeometryTerm(float VdH, float VdN, float alpha)
{
    float cVdH = clamp(VdH, .0, 1.);
    float chi = chiGGX(cVdH / clamp(VdN, .0, 1.));
    float tan2 = (1 - cVdH) / cVdH;
    return (chi * 2) / (1 + sqrt(1 + alpha * alpha * tan2));
}

// Disney's Principled BRDF
#define EPSILON .0001

uniform float sheen;
uniform float sheenTint;

vec3 calculateTint(vec3 baseColor)
{
    float luminance = dot(vec3(.3, .6, .1), baseColor);
    return luminance > .0 ? baseColor / luminance : vec3(1.);
}

float schlickWeight(float cosTheta)
{
    float m = clamp(1. - cosTheta, .0, 1.);
    return (m * m) * (m * m) * m;
}

vec3 evaluateSheen(float sheenF, float sheenTintF, vec3 baseColor, float dotHL)
{
    if(sheenF <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    float fh = schlickWeight(dotHL);
    return sheenF * mix(vec3(1.), tint, sheenTintF) * fh;
}

float GTR1(float dotHN, float alpha)
{
    if(alpha >= 1.)
        return 1. / PI;
    float alpha2 = alpha * alpha;
    float t = 1. + (alpha2 - 1.) * dotHN * dotHN;
    return (alpha2 - 1.) / (PI * log(alpha2) * t);
}

float separableSmithGGXG1(float absDotNV, float alpha)
{
    float alpha2 = alpha * alpha;
    return 2. / (1 + sqrt(alpha2 + (1 - alpha2) * absDotNV * absDotNV));
}

float smithGGGX(float dotNV, float alpha)
{
    float alpha2 = alpha * alpha;
    float b = dotNV * dotNV;
    return 1. / (abs(dotNV) + max(sqrt(alpha2 + b - alpha2 * b), EPSILON));
}
