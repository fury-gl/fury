bool inVisibleRange(vec3 center)
{
    bool xVal = lowRanges.x <= center.x && center.x <= highRanges.x;
    bool yVal = lowRanges.y <= center.y && center.y <= highRanges.y;
    bool zVal = lowRanges.z <= center.z && center.z <= highRanges.z;
    return xVal && yVal && zVal;
}
