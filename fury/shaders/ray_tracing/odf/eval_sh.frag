void evalSH(out float outSH[SH_COUNT], vec3 point, int shDegree, int numCoeffs)
{
    if (shDegree == 2)
    {
        float tmpOutSH[6];
        #if SH_DEGREE == 2
            evalSH2(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 4)
    {
        float tmpOutSH[15];
        #if SH_DEGREE == 4
            evalSH4(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 6)
    {
        float tmpOutSH[28];
        #if SH_DEGREE == 6
            evalSH6(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 8)
    {
        float tmpOutSH[45];
        #if SH_DEGREE == 8
            evalSH8(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 10)
    {
        float tmpOutSH[66];
        #if SH_DEGREE == 10
            evalSH10(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 12)
    {
        float tmpOutSH[91];
        #if SH_DEGREE == 12
            evalSH12(tmpOutSH, point);
        #endif
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
}
