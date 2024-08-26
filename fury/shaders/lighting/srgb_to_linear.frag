float srgb_to_linear(float non_linear)
{
    return (non_linear <= 0.04045) ? ((1.0 / 12.92) * non_linear) : pow(non_linear * (1.0 / 1.055) + 0.055 / 1.055, 2.4);
}
