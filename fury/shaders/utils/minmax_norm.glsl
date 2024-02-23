float coeffsNorm(float coef, float min, float max, float a, float b)
{
    return (coef - min) * ((b - a) / (max - min)) + a;
}