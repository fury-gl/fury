float srgbToLinear(float nonLinear)
{
    return (nonLinear <= 0.04045) ? ((1.0 / 12.92) * nonLinear) : pow(nonLinear * (1.0 / 1.055) + 0.055 / 1.055, 2.4);
}
