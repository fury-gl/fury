if (isRange)
{
    if (!inVisibleRange(centerVertexMCVSOutput))
        discard;
}
else
{
    if (!inVisibleCrossSection(centerVertexMCVSOutput))
        discard;
}
