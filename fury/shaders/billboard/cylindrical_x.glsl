vec3 cylindricalXVertexPos(vec3 center, mat4 MCVCMatrix, vec3 normalizedVertexMC, vec2 shape)
{
    vec3 cameraRightMC = vec3(1, 0, 0);
    vec3 cameraUpMC = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

    return center +
        cameraRightMC * shape.x * normalizedVertexMC.x +
        cameraUpMC * shape.y * normalizedVertexMC.y;
}
