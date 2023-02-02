float GTR1(float dotHN, float alpha)
{
    float a2 = square(alpha);
    float t = 1 + (a2 - 1) * square(dotHN);
    return (a2 - 1) / (PI * log(a2) * t);
}
