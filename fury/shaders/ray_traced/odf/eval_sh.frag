void evalSH(out float outSH[SH_COUNT], vec3 point, int shDegree, int numCoeffs)
{
    if (shDegree == 2)
    {
        float tmpOutSH[6];
        evalSH2(tmpOutSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 4)
    {
        float tmpOutSH[15];
        evalSH4(tmpOutSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 6)
    {
        float tmpOutSH[28];
        evalSH6(tmpOutSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 8)
    {
        float tmpOutSH[45];
        evalSH8(outSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 10)
    {
        float tmpOutSH[66];
        evalSH10(outSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
    else if (shDegree == 12)
    {
        float tmpOutSH[91];
        evalSH12(outSH, point);
        for (int i = 0; i != numCoeffs; ++i)
            outSH[i] = tmpOutSH[i];
    }
}
