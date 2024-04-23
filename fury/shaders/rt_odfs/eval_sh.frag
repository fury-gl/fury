void eval_sh(out float out_shs[SH_COUNT], vec3 point)
{
    #if SH_DEGREE == 2
        eval_sh_2(out_shs, point);
    #elif SH_DEGREE == 4
        eval_sh_4(out_shs, point);
    #elif SH_DEGREE == 6
        eval_sh_6(out_shs, point);
    #elif SH_DEGREE == 8
        eval_sh_8(out_shs, point);
    #elif SH_DEGREE == 10
        eval_sh_10(out_shs, point);
    #elif SH_DEGREE == 12
        eval_sh_12(out_shs, point);
    #endif
}
