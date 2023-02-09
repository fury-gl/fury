float smithGGXAnisotropic(float alphaX, float alphaY, float dotNV, float dotVX, float dotVY)
{
    float ax2 = square(alphaX);
    float ay2 = square(alphaY);
    float dotVX2 = square(dotVX);
    float dotVY2 = square(dotVY);
    return 1. / (dotNV + sqrt(dotVX2 * ax2 + dotVY2 * ay2 + square(dotNV)));
}
