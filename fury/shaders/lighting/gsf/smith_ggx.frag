float smithGGX(float alpha, float dotNV)
{
    float alpha2 = square(alpha);
    float b = square(dotNV);
    return 1. / (dotNV + sqrt(alpha2 + b - alpha2 * b));
}
