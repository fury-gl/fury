void evalSH(out float outSHs[SH_COUNT], vec3 point, int shDegree, int numCoeffs)
{
    if (shDegree == 2)
    {
        float tmpOutSHs[6];
        eval_sh_2(tmpOutSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
    else if (shDegree == 4)
    {
        float tmpOutSHs[15];
        eval_sh_4(tmpOutSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
    else if (shDegree == 6)
    {
        float tmpOutSHs[28];
        eval_sh_6(tmpOutSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
    else if (shDegree == 8)
    {
        float outSHs[45];
        eval_sh_8(outSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
    else if (shDegree == 10)
    {
        float outSHs[66];
        eval_sh_10(outSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
    else if (shDegree == 12)
    {
        float outSHs[91];
        eval_sh_12(outSHs, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSHs[i] = tmpOutSHs[i];
    }
}
