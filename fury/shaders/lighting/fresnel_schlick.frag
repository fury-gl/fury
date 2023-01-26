vec3 fresnelSchlick(vec3 f0, float dotHV)
{
    return f0 + (1 - f0) * pow(1 - dotHV, 5);
}
