void evalSH2(out float outSH[6], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    outSH[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    outSH[3] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    outSH[2] = -c1 * d;
    outSH[4] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    outSH[1] = c0 * d;
    outSH[5] = s0 * d;
}
