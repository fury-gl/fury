centerVertexMCVSOutput = center;
if (vertexColorVSOutput.rgb == vec3(0))
{
    vertexColorVSOutput.rgb = orient2rgb(diff);
}
