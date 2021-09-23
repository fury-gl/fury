float chiGGX(float v)
{
    return v > 0 ? 1. : .0;
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

uniform float subsurface;
uniform float specularTint;
uniform float anisotropic;
uniform float sheen;
uniform float sheenTint;
uniform float clearcoat;
uniform float clearcoatGloss;

float square(float x)
{
    return x * x;
}

vec3 calculateTint(vec3 baseColor)
{
    float luminance = dot(vec3(.3, .6, .1), baseColor);
    return luminance > .0 ? baseColor / luminance : vec3(1.);
}

void createBasis(vec3 normal, out vec3 tangent, out vec3 binormal)
{
    if(abs(normal.x) > abs(normal.y))
        tangent = normalize(vec3(.0, normal.z, -normal.y));
    else
        tangent = normalize(vec3(-normal.z, .0, normal.x));
    binormal = cross(normal, tangent);
}

float dielectric(float cosThetaI, float ni, float nt)
{
    cosThetaI = clamp(cosThetaI, -1., 1.);

    // Swap index of refraction if this is coming from inside the surface
    if(cosThetaI < .0)
    {
        float temp = ni;
        ni = nt;
        nt = temp;

        cosThetaI = -cosThetaI;
    }

    float sinThetaI = sqrt(max(.0, 1 - square(cosThetaI)));
    float sinThetaT = ni / nt * sinThetaI;

    // Check for total internal reflection
    if(sinThetaT >= 1.)
        return 1.;

    float cosThetaT = sqrt(max(.0, 1. - square(sinThetaT)));

    float rParallelNum = nt * cosThetaI - ni * cosThetaT;
    float rParallelDen = nt * cosThetaI + nt * cosThetaT;
    float rParallel = rParallelNum / rParallelDen;

    float rPerpendicularNum = ni * cosThetaI - nt * cosThetaT;
    float rPerpendicularDen = ni * cosThetaI + nt * cosThetaT;
    float rPerpendicular = rPerpendicularNum / rPerpendicularDen;

    return (square(rParallel) + square(rPerpendicular)) / 2;
}

void directionOfAnisotropicity(vec3 normal, out vec3 tangent, out vec3 binormal)
{
    //tangent = cross(normal, vec3(.5, .5, 1.));
    tangent = cross(normal, vec3(.0, 1., .5));
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    binormal = normalize(cross(normal, tangent));
    tangent = normalize(cross(normal, binormal));
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

float GGXAnisotropic(float dotHN, float dotHX, float dotHY, float ax, float ay)
{
    float a2 = ax * ay;
    vec3 d = vec3(ay * dotHX, ax * dotHY, a2 * dotHN);
    float d2 = dot(d, d);
    float b2 = a2 / d2;
    return a2 * square(b2) * recPI;
}

float schlickWeight(float cosTheta)
{
    float m = clamp(1. - cosTheta, .0, 1.);
    return m * m * m * m * m;
}

float schlickR0FromRelativeIOR(float eta)
{
    return square(eta - 1.) / square(eta + 1.);
}

vec3 fresnel(float specularTintF, float IORF, float relativeIORF,
             float metallicF, vec3 baseColor, float dotHL, float dotHV)
{
    vec3 tint = calculateTint(baseColor);

    /*
    See section 3.1 and 3.2 of the 2015 PBR presentation + the Disney BRDF
    explorer (which does their 2012 remapping rather than the
    SchlickR0FromRelativeIOR seen here but they mentioned the switch in 3.2).
    */
    vec3 tintMix = mix(vec3(1.), tint, specularTintF);
    vec3 r0 = schlickR0FromRelativeIOR(relativeIORF) * tintMix;
    r0 = mix(r0, baseColor, metallicF);

    float dielectricFresnel = dielectric(dotHV, 1., IORF);
    vec3 metallicFresnel = F_Schlick(dotHL, r0);

    return mix(vec3(dielectricFresnel), metallicFresnel, metallicF);
}

float separableSmithGGXG1(float dotNV, float alpha)
{
    float alpha2 = square(alpha);
    return 2. / (1 + sqrt(alpha2 + (1 - alpha2) * dotNV * dotNV));
}

float separableSmithGGXG1(float dotVX, float dotVY, float dotNV,
                          float ax, float ay)
{
    float dotVX2 = square(dotVX);
    float dotVY2 = square(dotVY);
    float ax2 = square(ax);
    float ay2 = square(ay);

    float absTanTheta = abs(dotNV);
    if(isinf(absTanTheta))
        return .0;

    float alpha = absTanTheta * sqrt(dotVX2 * ax2 + dotVY2 * ay2);
    float alpha2 = square(alpha);

    float lambda = .5 * (-1. + sqrt(1. + alpha2));

    return 1. / (1. + lambda);
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

float smithGGGXCorrelatedAnisotropic(float dotLN, float dotLX, float dotLY,
                                     float dotNV, float dotVX, float dotVY,
                                     float ax, float ay)
{
    float lambdaL = dotNV * length(vec3(ax * dotLX, ay * dotLY, dotLN));
    float lambdaV = dotLN * length(vec3(ax * dotVX, ay * dotVY, dotNV));
    return .5 / (lambdaL + lambdaV);
}

vec3 evaluateBRDF(float anisotropicF, float roughnessF, float specularTintF,
                  float IORF, float relativeIORF, float metallicF,
                  vec3 baseColor, float dotHL, float dotHN, float dotHV,
                  float dotHX, float dotHY, float dotLN, float dotLX,
                  float dotLY, float dotNV, float dotVX, float dotVY)
{
    if(dotLN <= .0 || dotNV <= .0)
        return vec3(.0);

    float aspect = sqrt(1. - anisotropicF * .9);

    float ax = max(.001, square(roughnessF) / aspect);
    float ay = max(.001, square(roughnessF) * aspect);

    float d = GTR2Anisotropic(dotHN, dotHX, dotHY, ax, ay);
    float gl = separableSmithGGXG1(dotLX, dotLY, dotLN, ax, ay);
    float gv = separableSmithGGXG1(dotVX, dotVY, dotNV, ax, ay);

    vec3 f = fresnel(specularTintF, IORF, relativeIORF, metallicF, baseColor,
        dotHL, dotHV);

    return d * gl * gv * f / (4. * dotLN * dotNV);
}

float evaluateClearcoat(float clearcoatF, float clearcoatGlossF, float dotHL,
                        float dotHN, float dotLN, float dotNV)
{
    if(clearcoatF <= .0)
        return .0;

    float gloss = mix(.1, .001, clearcoatGlossF);
    float dr = GTR1(abs(dotHN), gloss);
    float fh = schlickWeight(dotHL);
    float fr = mix(.04, 1., fh);
    float gr = smithGGGX(dotLN, .25) * smithGGGX(dotNV, .25);
    return 1. * clearcoatF * fr * gr * dr;

    float gl = separableSmithGGXG1(abs(dotLN), .25);
    float gv = separableSmithGGXG1(abs(dotNV), .25);
    //float gl = separableSmithGGXG1(dotLN, .25);
    //float gv = separableSmithGGXG1(dotNV, .25);
    return .25 * clearcoatF * dr * fr * gl * gv;
}

vec3 evaluateDiffuse(float roughnessF, vec3 baseColor, float dotHL,
                     float dotLN, float dotNV)
{
    float fl = schlickWeight(dotLN);
    float fv = schlickWeight(dotNV);

    float fd90 = .5 + 2. * square(dotHL) * roughnessF;
    float fd = mix(1., fd90, fl) * mix(1., fd90, fv);
    return (1. / PI) * fd * baseColor;
}

vec3 evaluateMicrofacetAnisotropic(float specularF, float specularTintF,
                                   float metallicF, float anisotropicF,
                                   float roughnessF, vec3 baseColor,
                                   float dotHL, float dotHN, float dotHX,
                                   float dotHY, float dotLN, float dotLX,
                                   float dotLY, float dotNV, float dotVX,
                                   float dotVY)
{
    if(dotLN <= .0 || dotNV <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    vec3 tintMix = mix(vec3(1.), tint, specularTintF);
    vec3 spec = mix(specularF * .08 * tintMix, baseColor, metallicF);

    float aspect = sqrt(1. - anisotropicF * .9);

    float ax = max(.001, square(roughnessF) / aspect);
    float ay = max(.001, square(roughnessF) * aspect);

    float ds = GTR2Anisotropic(dotHN, dotHX, dotHY, ax, ay);

    //float ds = GGXAnisotropic(dotHN, dotHX, dotHY, ax, ay);

    float fh = schlickWeight(dotHL);
    vec3 fs = mix(spec, vec3(1.), fh);

    float gs = smithGGGXAnisotropic(dotLN, dotLX, dotLY, ax, ay);
    gs *= smithGGGXAnisotropic(dotNV, dotVX, dotVY, ax, ay);

    //float gs = smithGGGXCorrelatedAnisotropic(dotLN, dotLX, dotLY, dotNV,
        //dotVX, dotVY, ax, ay);

    return gs * fs * ds;
}

vec3 evaluateMicrofacetIsotropic(float specularF, float specularTintF,
                                 float metallicF, float roughnessF,
                                 vec3 baseColor, float dotHL, float dotHN,
                                 float dotLN, float dotNV)
{
    if(specularF <= 0)
        return vec3(.0);
    if(dotLN <= .0 || dotNV <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    vec3 tintMix = mix(vec3(1.), tint, specularTintF);
    vec3 spec = mix(specularF * .08 * tintMix, baseColor, metallicF);

    float a = max(.001, square(roughnessF));

    float ds = GTR2(dotHN, a);
    float fh = schlickWeight(dotHL);
    vec3 fs = mix(spec, vec3(1.), fh);
    float gs = smithGGGX(dotLN, a);
    gs *= smithGGGX(dotNV, a);
    return gs * fs * ds;
}

vec3 vtkSpecularAnisotropic(float anisotropicF, float roughnessF, float dotHL,
                            float dotHN, float dotHX, float dotHY, float dotLN,
                            float dotLX, float dotLY, float dotNV, float dotVX,
                            float dotVY, vec3 F0, vec3 F90)
{
    float aspect = sqrt(1. - anisotropicF * .9);

    float ax = max(.001, square(roughnessF) / aspect);
    float ay = max(.001, square(roughnessF) * aspect);

    float d = GGXAnisotropic(dotHN, dotHX, dotHY, ax, ay);

    float g = smithGGGXCorrelatedAnisotropic(dotLN, dotLX, dotLY, dotNV, dotVX,
        dotVY, ax, ay);

    vec3 f = F0 + (F90 - F0) * schlickWeight(dotHL);

    return (d * g) * f;
}

vec3 evaluateSheen(float sheenF, float sheenTintF, vec3 baseColor, float dotHL)
{
    if(sheenF <= .0)
        return vec3(.0);
    vec3 tint = calculateTint(baseColor);
    float fh = schlickWeight(dotHL);
    vec3 tintMix = mix(vec3(1.), tint, sheenTintF);
    return sheenF * tintMix * fh;
}

vec3 evaluateSubsurface(float roughnessF, vec3 baseColor, float dotLN,
                        float dotNV, float dotHL)
{
    float fl = schlickWeight(dotLN);
    float fv = schlickWeight(dotNV);

    float fss90 = square(dotHL) * roughnessF;
    float fss = mix(1., fss90, fl) * mix(1., fss90, fv);
    float ss = 1.25 * (fss * (1. / (dotLN + dotNV) - .5) + .5);
    return (1 / PI) * ss * baseColor;
}
