float schlickWeight(float dotHV)
{
    return pow5(clamp(1 - dotHV, 0, 1));
}
