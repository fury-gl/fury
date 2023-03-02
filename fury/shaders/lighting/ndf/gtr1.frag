float GTR1(float alpha, float dotHN)
{
    float a2 = square(alpha);
    float t = 1 + (a2 - 1) * square(dotHN);
    return (a2 - 1) / (PI * log(a2) * t);
}
