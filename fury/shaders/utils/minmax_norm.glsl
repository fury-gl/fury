float rescale(float x, float oldMin, float oldMax, float newMin, float newMax)
{
    return (x - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
}
