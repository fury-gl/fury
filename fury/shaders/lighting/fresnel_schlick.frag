vec3 fresnelSchlick(vec3 f0, float dotHV)
{
    return f0 + (1 - f0) * pow5(1 - dotHV);
}
