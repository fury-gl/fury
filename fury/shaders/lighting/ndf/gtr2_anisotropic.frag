float GTR2Anisotropic(float alphaX, float alphaY, float dotHN, float dotHX,
    float dotHY)
{
    float ax2 = square(alphaX);
    float ay2 = square(alphaY);
    float t = square(dotHX) / ax2 + square(dotHY) / ay2 + square(dotHN);
    return 1 / (PI * ax * ay * square(t));
}
