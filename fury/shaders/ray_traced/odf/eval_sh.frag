void evalSH(out float outSHs[SH_COUNT], vec3 point, int shDegree)
{
    #if SH_DEGREE == 2
        eval_sh_2(outSHs, point);
    #elif SH_DEGREE == 4
        eval_sh_4(outSHs, point);
    #elif SH_DEGREE == 6
        eval_sh_6(outSHs, point);
    #elif SH_DEGREE == 8
        eval_sh_8(outSHs, point);
    #elif SH_DEGREE == 10
        eval_sh_10(outSHs, point);
    #elif SH_DEGREE == 12
        eval_sh_12(outSHs, point);
    #endif
}
