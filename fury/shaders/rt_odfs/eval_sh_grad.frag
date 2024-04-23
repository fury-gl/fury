void eval_sh_grad(out float out_shs[SH_COUNT], out vec3 out_grads[SH_COUNT], vec3 point)
{
    #if SH_DEGREE == 2
        eval_sh_grad_2(out_shs, out_grads, point);
    #elif SH_DEGREE == 4
        eval_sh_grad_4(out_shs, out_grads, point);
    #elif SH_DEGREE == 6
        eval_sh_grad_6(out_shs, out_grads, point);
    #elif SH_DEGREE == 8
        eval_sh_grad_8(out_shs, out_grads, point);
    #elif SH_DEGREE == 10
        eval_sh_grad_10(out_shs, out_grads, point);
    #elif SH_DEGREE == 12
        eval_sh_grad_12(out_shs, out_grads, point);
    #endif
}
