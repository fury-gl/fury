in vec3 centerVertexMCVSOutput;

uniform bool isRange;
uniform vec3 crossSection;
uniform vec3 lowRanges;
uniform vec3 highRanges;

bool inVisibleCrossSection(vec3 center)
{
    bool xVal = center.x == crossSection.x;
    bool yVal = center.y == crossSection.y;
    bool zVal = center.z == crossSection.z;
    return xVal || yVal || zVal;
}

bool inVisibleRange(vec3 center)
{
    bool xVal = lowRanges.x <= center.x && center.x <= highRanges.x;
    bool yVal = lowRanges.y <= center.y && center.y <= highRanges.y;
    bool zVal = lowRanges.z <= center.z && center.z <= highRanges.z;
    return xVal && yVal && zVal;
}
