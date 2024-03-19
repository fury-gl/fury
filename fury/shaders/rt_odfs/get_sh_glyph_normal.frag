vec3 get_sh_glyph_normal(float sh_coeffs[SH_COUNT], vec3 point)
{
    float shs[SH_COUNT];
    vec3 grads[SH_COUNT];
    float length_inv = inversesqrt(dot(point, point));
    vec3 normalized = point * length_inv;
    eval_sh_grad(shs, grads, normalized);
    float value = 0.0;
    vec3 grad = vec3(0.0);
    _unroll_
    for (int i = 0; i != SH_COUNT; ++i) {
        value += sh_coeffs[i] * shs[i];
        grad += sh_coeffs[i] * grads[i];
    }
    return normalize(point - (value * length_inv) * (grad - dot(grad, normalized) * normalized));
}
