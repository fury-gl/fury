void eval_sh_4(out float outSH[15], vec3 point)
{
    float x, y, z, z2, c0, s0, c1, s1, d, a, b;
    x = point[0];
    y = point[1];
    z = point[2];
    z2 = z * z;
    d = 0.282094792;
    outSH[0] = d;
    a = z2 - 0.333333333;
    d = 0.946174696 * a;
    outSH[3] = d;
    b = z2 * (a - 0.266666667);
    a = b - 0.257142857 * a;
    d = 3.702494142 * a;
    outSH[10] = d;
    c1 = x;
    s1 = y;
    d = -1.092548431 * z;
    outSH[4] = c1 * d;
    outSH[2] = s1 * d;
    a = (z2 - 0.2) * z;
    b = a - 0.228571429 * z;
    d = -4.683325805 * b;
    outSH[11] = c1 * d;
    outSH[9] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.546274215;
    outSH[5] = c0 * d;
    outSH[1] = s0 * d;
    a = z2 - 0.142857143;
    d = 3.311611435 * a;
    outSH[12] = c0 * d;
    outSH[8] = s0 * d;
    c1 = x * c0 - y * s0;
    s1 = y * c0 + x * s0;
    d = -1.77013077 * z;
    outSH[13] = c1 * d;
    outSH[7] = s1 * d;
    c0 = x * c1 - y * s1;
    s0 = y * c1 + x * s1;
    d = 0.625835735;
    outSH[14] = c0 * d;
    outSH[6] = s0 * d;
}
