float GTR2(float dotHN, float alpha)
{
    float a2 = square(alpha);
    float t = 1 + (a2 - 1) * square(dotHN);
    return a2 / (PI * square(t));
}
