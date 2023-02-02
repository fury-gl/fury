float GTR(float dotHN, float alpha, float gamma)
{
    float a2 = square(alpha);
    float t = 1 + (a2 - 1) * square(dotHN);
    return ((gamma - 1) * (a2 - 1)) / (PI * (1 - pow(a2, (1 - gamma))) * pow(t, gamma));
}
