vec3 cylindricalYVertexPos(vec3 center, mat4 MCVCMatrix, vec3 normalizedVertexMC, vec2 shape)
{
    vec3 cameraRightMC = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
    vec3 cameraUpMC = vec3(0, 1, 0);

    return center +
        cameraRightMC * shape.x * normalizedVertexMC.x +
        cameraUpMC * shape.y * normalizedVertexMC.y;
}
