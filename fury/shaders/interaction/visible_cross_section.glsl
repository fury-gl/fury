bool inVisibleCrossSection(ivec3 center)
{
    bool xVal = center.x == crossSection.x;
    bool yVal = center.y == crossSection.y;
    bool zVal = center.z == crossSection.z;
    return xVal || yVal || zVal;
}
