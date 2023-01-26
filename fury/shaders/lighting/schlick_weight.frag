float schlickWeight(float dotHV)
{
    float w = clamp(1 - dotHV, 0, 1);
    return w * w * w * w * w;
}
