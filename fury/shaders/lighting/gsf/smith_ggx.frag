float smithGGGX(float alpha, float dotNV)
{
    float alpha2 = square(alpha);
    float b = square(dotNV);
    return 1. / (abs(dotNV) + max(sqrt(alpha2 + b - alpha2 * b), EPSILON));
}
