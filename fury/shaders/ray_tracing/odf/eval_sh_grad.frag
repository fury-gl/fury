void evalShGrad(out float outSH[SH_COUNT], out vec3 outGrads[SH_COUNT], vec3 point, int shDegree,
        int numCoeffs)
{
    if (shDegree == 2)
    {
        float tmpOutSH[6];
        float tmpOutGrads[6];
        #if SH_DEGREE == 2
            evalShGrad2(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];

    }
    else if (shDegree == 4)
    {
        float tmpOutSH[15];
        #if SH_DEGREE == 4
            evalShGrad4(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 6)
    {
        float tmpOutSH[28];
        #if SH_DEGREE == 6
            evalShGrad6(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 8)
    {
        float tmpOutSH[45];
        #if SH_DEGREE == 8
            evalShGrad8(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 10)
    {
        float tmpOutSH[66];
        #if SH_DEGREE == 10
            evalShGrad10(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 12)
    {
        float tmpOutSH[91];
        #if SH_DEGREE == 12
            evalShGrad12(tmpOutSH, outGrads, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
}
